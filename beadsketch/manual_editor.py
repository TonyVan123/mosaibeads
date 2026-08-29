from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .engine import PatternResult
from .exporter import export_bundle
from .palettes import BeadColor, load_palette, resource_root


BUILTIN_PROJECTS = {
    "原始识别版（58×57）": "couple_original_58x57.json",
    "脸型/吊带/右眼调整版（58×57）": "couple_face_strap_revision_58x57.json",
}


@dataclass
class LoadedProject:
    result: PatternResult
    raw: dict
    source_path: Path


def load_project(path: str | Path) -> LoadedProject:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("项目文件顶层必须是对象。")
    grid = data.get("grid")
    indices = np.asarray(data.get("indices"), dtype=np.int32)
    if indices.ndim != 2 or not indices.size:
        raise ValueError("项目中的 indices 必须是非空二维数组。")
    height, width = indices.shape
    if grid and [int(grid[0]), int(grid[1])] != [width, height]:
        raise ValueError(f"项目标注尺寸 {grid} 与格子数据 {width}×{height} 不一致。")

    palette_data = data.get("palette") or []
    palette: list[BeadColor] = []
    for item in palette_data:
        rgb = tuple(int(v) for v in item["rgb"])
        if len(rgb) != 3 or any(v < 0 or v > 255 for v in rgb):
            raise ValueError("色板中存在无效 RGB。")
        palette.append(BeadColor(str(item["code"]), str(item.get("name") or item["code"]), rgb))
    if not palette:
        raise ValueError("项目色板为空。")
    if int(indices.min()) < 0 or int(indices.max()) >= len(palette):
        raise ValueError("格子中引用了色板范围以外的颜色。")

    brand = str((data.get("metadata") or {}).get("brand_palette") or "MARD 291")
    seen_codes = {c.code.upper() for c in palette}
    try:
        for color in load_palette(brand):
            if color.code.upper() not in seen_codes:
                palette.append(color)
                seen_codes.add(color.code.upper())
    except Exception:
        pass
    colors = np.asarray([c.rgb for c in palette], dtype=np.uint8)
    rgb_grid = colors[indices]
    result = PatternResult(
        indices=indices.copy(),
        palette=palette,
        selected_source_indices=np.arange(len(palette), dtype=np.int32),
        source_rgb=rgb_grid.copy(),
        sampled_rgb=rgb_grid.copy(),
        saliency=np.zeros((height, width), dtype=np.float32),
        profile=str(data.get("profile") or "手工编辑"),
        metadata=dict(data.get("metadata") or {}),
    )
    return LoadedProject(result=result, raw=data, source_path=path)


def project_dict(result: PatternResult, base: dict | None = None) -> dict:
    data = dict(base or {})
    metadata = dict(data.get("metadata") or {})
    metadata.update(
        {
            "bead_count": result.width * result.height,
            "mapped_colors": len(result.counts()),
            "last_edited_by": "MOSAIBeads Manual Editor",
            "last_edited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    data.update(
        {
            "app": "MOSAIBeads Manual Editor",
            "grid": [result.width, result.height],
            "profile": result.profile,
            "metadata": metadata,
            "palette": [
                {"code": c.code, "name": c.name, "rgb": list(c.rgb)} for c in result.palette
            ],
            "indices": result.indices.tolist(),
        }
    )
    return data


def save_project(path: str | Path, result: PatternResult, base: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(project_dict(result, base), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def builtin_project_path(label: str) -> Path:
    return resource_root() / "assets" / "projects" / BUILTIN_PROJECTS[label]


class ManualEditor:
    BG = "#111827"
    PANEL = "#1F2937"
    PANEL_2 = "#273449"
    TEXT = "#F9FAFB"
    MUTED = "#AAB5C5"
    ACCENT = "#FF9F43"
    ACCENT_DARK = "#C96B17"
    CANVAS_BG = "#E8ECF1"
    GRID = "#687383"
    GRID_5 = "#354154"
    GRID_29 = "#F15B5B"
    ZOOMS = (10, 12, 16, 20, 24, 30, 36, 44)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MOSAIBeads 手工拼豆编辑器")
        self.root.geometry("1450x900")
        self.root.minsize(980, 650)
        self.root.configure(bg=self.BG)

        self.loaded: LoadedProject | None = None
        self.result: PatternResult | None = None
        self.current_path: Path | None = None
        self.selected_palette = 0
        self.edit_mode_var = tk.StringVar(value="paint")
        self.palette_filter_var = tk.StringVar()
        self.used_only_var = tk.BooleanVar(value=False)
        self.zoom_index = 4
        self.margin = 42
        self.dirty = False
        self._loading_version = False
        self._last_builtin = next(iter(BUILTIN_PROJECTS))
        self._stroke: dict[tuple[int, int], tuple[int, int]] = {}
        self.undo_stack: list[list[tuple[int, int, int, int]]] = []
        self.redo_stack: list[list[tuple[int, int, int, int]]] = []
        self.cell_rects: list[list[int]] = []
        self.cell_texts: list[list[int | None]] = []
        self.palette_rows: list[tk.Frame] = []
        self.palette_row_indices: list[int] = []
        self.selection: tuple[int, int, int, int] | None = None
        self.selection_anchor: tuple[int, int] | None = None
        self.selection_drag_kind = "cells"
        self.selection_rect: int | None = None
        self.clipboard_codes: list[list[str]] = []

        self._configure_style()
        self._build_ui()
        self._bind_shortcuts()
        self.load_builtin(self._last_builtin, confirm=False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    @property
    def cell(self) -> int:
        return self.ZOOMS[self.zoom_index]

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dark.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Dark.TLabel", background=self.BG, foreground=self.TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(12, 7))
        style.configure("Tool.TButton", font=("Microsoft YaHei UI", 9), padding=(8, 6))
        style.configure("TCombobox", padding=5, font=("Microsoft YaHei UI", 10))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="Dark.TFrame", padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="MOSAIBeads 手工编辑器", style="Title.TLabel").pack(side="left", padx=(0, 20))

        ttk.Label(header, text="内置版本", style="Dark.TLabel").pack(side="left", padx=(0, 7))
        self.version_var = tk.StringVar(value=self._last_builtin)
        self.version_box = ttk.Combobox(
            header,
            textvariable=self.version_var,
            values=list(BUILTIN_PROJECTS),
            width=31,
            state="readonly",
        )
        self.version_box.pack(side="left", padx=(0, 9))
        self.version_box.bind("<<ComboboxSelected>>", self._on_version_selected)
        ttk.Button(header, text="打开项目", style="Tool.TButton", command=self.open_project).pack(side="left", padx=3)
        ttk.Button(header, text="保存项目", style="Tool.TButton", command=self.save_current).pack(side="left", padx=3)
        ttk.Button(header, text="导出图纸", style="Accent.TButton", command=self.export_current).pack(side="left", padx=(8, 3))

        toolbar = ttk.Frame(self.root, style="Panel.TFrame", padding=(14, 7))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="↶ 撤销", style="Tool.TButton", command=self.undo).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="↷ 重做", style="Tool.TButton", command=self.redo).pack(side="left", padx=5)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Radiobutton(toolbar, text="画笔", variable=self.edit_mode_var, value="paint", command=self._mode_changed).pack(side="left", padx=3)
        ttk.Radiobutton(toolbar, text="矩形选择", variable=self.edit_mode_var, value="select", command=self._mode_changed).pack(side="left", padx=3)
        ttk.Button(toolbar, text="填充选区", style="Accent.TButton", command=self.fill_selection).pack(side="left", padx=(7, 3))
        ttk.Button(toolbar, text="整行", style="Tool.TButton", command=self.select_whole_row).pack(side="left", padx=3)
        ttk.Button(toolbar, text="整列", style="Tool.TButton", command=self.select_whole_column).pack(side="left", padx=3)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Button(toolbar, text="－", width=3, style="Tool.TButton", command=lambda: self.change_zoom(-1)).pack(side="left")
        self.zoom_label = ttk.Label(toolbar, text="24 px/格", style="Panel.TLabel", width=10, anchor="center")
        self.zoom_label.pack(side="left", padx=5)
        ttk.Button(toolbar, text="＋", width=3, style="Tool.TButton", command=lambda: self.change_zoom(1)).pack(side="left")
        ttk.Button(toolbar, text="适合窗口", style="Tool.TButton", command=self.fit_view).pack(side="left", padx=(7, 0))
        ttk.Label(
            toolbar,
            text="画笔拖动上色 · 选择模式框选 · 右键取色 · 滚轮移动",
            style="Panel.TLabel",
        ).pack(side="right", padx=8)

        body = ttk.Frame(self.root, style="Dark.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(body, style="Dark.TFrame", padding=(12, 12, 6, 8))
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg=self.CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._left_motion)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)
        self.canvas.bind("<Button-3>", self._pick_color)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._shift_mousewheel)

        side = tk.Frame(body, bg=self.PANEL, width=310)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)

        info = tk.Frame(side, bg=self.PANEL, padx=14, pady=12)
        info.pack(fill="x")
        tk.Label(info, text="当前颜色", bg=self.PANEL, fg=self.MUTED, font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        self.current_color = tk.Frame(info, bg=self.PANEL_2, highlightthickness=3, highlightbackground=self.ACCENT, padx=8, pady=8)
        self.current_color.pack(fill="x", pady=(5, 10))
        self.current_swatch = tk.Label(self.current_color, width=4, height=2, bg="#FFFFFF", relief="solid", bd=1)
        self.current_swatch.pack(side="left", padx=(0, 10))
        self.current_text = tk.Label(self.current_color, text="", bg=self.PANEL_2, fg=self.TEXT, font=("Microsoft YaHei UI", 11, "bold"), anchor="w")
        self.current_text.pack(side="left", fill="x", expand=True)
        self.summary_label = tk.Label(info, text="", bg=self.PANEL, fg=self.MUTED, justify="left", anchor="w", font=("Microsoft YaHei UI", 9))
        self.summary_label.pack(fill="x")

        tk.Label(side, text="完整品牌色板（点击选择）", bg=self.PANEL, fg=self.TEXT, font=("Microsoft YaHei UI", 11, "bold"), padx=14, pady=8).pack(fill="x", anchor="w")
        palette_tools = tk.Frame(side, bg=self.PANEL, padx=12, pady=3)
        palette_tools.pack(fill="x")
        search = tk.Entry(palette_tools, textvariable=self.palette_filter_var, bg="#F7F9FC", fg="#111827", font=("Microsoft YaHei UI", 9))
        search.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)
        search.insert(0, "")
        self.palette_filter_var.trace_add("write", lambda *_args: self._render_palette())
        tk.Checkbutton(palette_tools, text="只看已用", variable=self.used_only_var, command=self._render_palette,
                       bg=self.PANEL, fg=self.TEXT, selectcolor=self.PANEL_2, activebackground=self.PANEL,
                       activeforeground=self.TEXT, font=("Microsoft YaHei UI", 9)).pack(side="right")
        palette_holder = tk.Frame(side, bg=self.PANEL)
        palette_holder.pack(fill="both", expand=True, padx=(8, 2), pady=(0, 8))
        self.palette_canvas = tk.Canvas(palette_holder, bg=self.PANEL, highlightthickness=0)
        palette_scroll = ttk.Scrollbar(palette_holder, orient="vertical", command=self.palette_canvas.yview)
        self.palette_canvas.configure(yscrollcommand=palette_scroll.set)
        palette_scroll.pack(side="right", fill="y")
        self.palette_canvas.pack(side="left", fill="both", expand=True)
        self.palette_inner = tk.Frame(self.palette_canvas, bg=self.PANEL)
        self.palette_window = self.palette_canvas.create_window((0, 0), window=self.palette_inner, anchor="nw")
        self.palette_inner.bind("<Configure>", lambda _e: self.palette_canvas.configure(scrollregion=self.palette_canvas.bbox("all")))
        self.palette_canvas.bind("<Configure>", lambda e: self.palette_canvas.itemconfigure(self.palette_window, width=e.width))
        self.palette_canvas.bind("<MouseWheel>", lambda e: self.palette_canvas.yview_scroll(-int(e.delta / 120), "units"))

        status = tk.Frame(self.root, bg="#0B1220", padx=12, pady=5)
        status.pack(fill="x")
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(status, textvariable=self.status_var, bg="#0B1220", fg=self.MUTED, font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-z>", lambda _e: self.undo())
        self.root.bind_all("<Control-y>", lambda _e: self.redo())
        self.root.bind_all("<Control-s>", lambda _e: self.save_current())
        self.root.bind_all("<Control-o>", lambda _e: self.open_project())
        self.root.bind_all("<Control-plus>", lambda _e: self.change_zoom(1))
        self.root.bind_all("<Control-minus>", lambda _e: self.change_zoom(-1))
        self.root.bind_all("<Control-c>", lambda _e: self.copy_selection())
        self.root.bind_all("<Control-v>", lambda _e: self.paste_selection())
        self.root.bind_all("<Control-a>", lambda _e: self.select_all())
        self.root.bind_all("<Return>", lambda _e: self.fill_selection())

    def _on_version_selected(self, _event: tk.Event | None = None) -> None:
        if self._loading_version:
            return
        label = self.version_var.get()
        if not self.load_builtin(label, confirm=True):
            self._loading_version = True
            self.version_var.set(self._last_builtin)
            self._loading_version = False

    def _confirm_replace(self) -> bool:
        if not self.dirty:
            return True
        choice = messagebox.askyesnocancel("尚未保存", "当前修改尚未保存。是否先保存项目？", parent=self.root)
        if choice is None:
            return False
        if choice:
            return self.save_current()
        return True

    def load_builtin(self, label: str, confirm: bool = True) -> bool:
        if confirm and not self._confirm_replace():
            return False
        try:
            loaded = load_project(builtin_project_path(label))
        except Exception as exc:
            messagebox.showerror("无法载入内置版本", str(exc), parent=self.root)
            return False
        self._last_builtin = label
        self._loading_version = True
        self.version_var.set(label)
        self._loading_version = False
        self._install_project(loaded, shown_name=label)
        return True

    def _install_project(self, loaded: LoadedProject, shown_name: str | None = None) -> None:
        self.loaded = loaded
        self.result = loaded.result
        self.current_path = None
        self.selected_palette = 0
        self.selection = None
        self.selection_anchor = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self._render_palette()
        self._render_grid()
        self._refresh_summary()
        self._set_title(shown_name or loaded.source_path.stem)
        self.status_var.set(f"已载入：{shown_name or loaded.source_path.name}")

    def _set_title(self, name: str) -> None:
        mark = " *" if self.dirty else ""
        self.root.title(f"MOSAIBeads 手工拼豆编辑器 — {name}{mark}")

    def _project_name(self) -> str:
        if self.current_path:
            return self.current_path.stem
        return self.version_var.get() or "未命名项目"

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._set_title(self._project_name())

    def _render_grid(self, keep_center: tuple[float, float] | None = None) -> None:
        if self.result is None:
            return
        if keep_center is None:
            keep_center = (self.canvas.canvasx(max(0, self.canvas.winfo_width() / 2)), self.canvas.canvasy(max(0, self.canvas.winfo_height() / 2)))
        old_total_w = max(1, self.margin * 2 + self.result.width * self.cell)
        old_total_h = max(1, self.margin * 2 + self.result.height * self.cell)
        center_ratio = (keep_center[0] / old_total_w, keep_center[1] / old_total_h)

        self.canvas.delete("all")
        self.cell_rects = []
        self.cell_texts = []
        cell = self.cell
        show_codes = cell >= 20
        font_size = max(7, min(11, round(cell * 0.34)))
        for y in range(self.result.height):
            rect_row: list[int] = []
            text_row: list[int | None] = []
            for x in range(self.result.width):
                idx = int(self.result.indices[y, x])
                color = self.result.palette[idx]
                x0 = self.margin + x * cell
                y0 = self.margin + y * cell
                rect = self.canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=self._hex(color.rgb), outline="")
                rect_row.append(rect)
                text_id: int | None = None
                if show_codes:
                    text_id = self.canvas.create_text(
                        x0 + cell / 2,
                        y0 + cell / 2,
                        text=color.code,
                        fill=self._text_hex(color.rgb),
                        font=("Arial", font_size, "bold"),
                    )
                text_row.append(text_id)
            self.cell_rects.append(rect_row)
            self.cell_texts.append(text_row)

        width_px = self.result.width * cell
        height_px = self.result.height * cell
        for x in range(self.result.width + 1):
            px = self.margin + x * cell
            color, line_w = (self.GRID_29, 3) if x % 29 == 0 else ((self.GRID_5, 2) if x % 5 == 0 else (self.GRID, 1))
            self.canvas.create_line(px, self.margin, px, self.margin + height_px, fill=color, width=line_w)
        for y in range(self.result.height + 1):
            py = self.margin + y * cell
            color, line_w = (self.GRID_29, 3) if y % 29 == 0 else ((self.GRID_5, 2) if y % 5 == 0 else (self.GRID, 1))
            self.canvas.create_line(self.margin, py, self.margin + width_px, py, fill=color, width=line_w)
        coord_font = ("Microsoft YaHei UI", 8)
        step = 5 if cell >= 16 else 10
        for x in range(0, self.result.width, step):
            self.canvas.create_text(self.margin + x * cell + cell / 2, self.margin - 15, text=str(x + 1), fill="#283444", font=coord_font)
        for y in range(0, self.result.height, step):
            self.canvas.create_text(self.margin - 19, self.margin + y * cell + cell / 2, text=str(y + 1), fill="#283444", font=coord_font)
        total_w = self.margin * 2 + width_px
        total_h = self.margin * 2 + height_px
        self.canvas.configure(scrollregion=(0, 0, total_w, total_h))
        self.canvas.xview_moveto(max(0.0, min(1.0, center_ratio[0] - self.canvas.winfo_width() / max(1, total_w) / 2)))
        self.canvas.yview_moveto(max(0.0, min(1.0, center_ratio[1] - self.canvas.winfo_height() / max(1, total_h) / 2)))
        self.zoom_label.configure(text=f"{cell} px/格")
        self._draw_selection()

    def _render_palette(self) -> None:
        for widget in self.palette_inner.winfo_children():
            widget.destroy()
        self.palette_rows.clear()
        self.palette_row_indices.clear()
        if self.result is None:
            return
        counts = {color.code: count for color, count in self.result.counts()}
        query = self.palette_filter_var.get().strip().lower()
        for idx, color in enumerate(self.result.palette):
            if self.used_only_var.get() and counts.get(color.code, 0) == 0:
                continue
            if query and query not in color.code.lower() and query not in color.name.lower():
                continue
            frame = tk.Frame(self.palette_inner, bg=self.PANEL_2, highlightthickness=3, highlightbackground=self.PANEL, padx=6, pady=5)
            frame.pack(fill="x", padx=5, pady=3)
            mark = tk.Label(frame, text="✓" if idx == self.selected_palette else "", width=2, bg=self.PANEL_2, fg=self.ACCENT, font=("Arial", 14, "bold"))
            mark.pack(side="left")
            swatch = tk.Label(frame, bg=self._hex(color.rgb), width=4, height=2, relief="solid", bd=1)
            swatch.pack(side="left", padx=(2, 8))
            label = tk.Label(frame, text=f"{color.code}\n{color.name}", bg=self.PANEL_2, fg=self.TEXT, justify="left", anchor="w", font=("Microsoft YaHei UI", 9, "bold"))
            label.pack(side="left", fill="x", expand=True)
            count = tk.Label(frame, text=str(counts.get(color.code, 0)), bg=self.PANEL_2, fg=self.MUTED, font=("Microsoft YaHei UI", 9))
            count.pack(side="right", padx=5)
            for widget in (frame, mark, swatch, label, count):
                widget.bind("<Button-1>", lambda _e, i=idx: self.select_color(i))
                widget.bind("<MouseWheel>", lambda e: self.palette_canvas.yview_scroll(-int(e.delta / 120), "units"))
            self.palette_rows.append(frame)
            self.palette_row_indices.append(idx)
        self._refresh_palette_selection()

    def _refresh_palette_selection(self) -> None:
        if self.result is None:
            return
        for row, palette_idx in zip(self.palette_rows, self.palette_row_indices):
            selected = palette_idx == self.selected_palette
            row.configure(highlightbackground=self.ACCENT if selected else self.PANEL, bg=self.PANEL_2)
            children = row.winfo_children()
            if children:
                children[0].configure(text="✓" if selected else "")
        color = self.result.palette[self.selected_palette]
        self.current_swatch.configure(bg=self._hex(color.rgb))
        label = color.code if color.name == color.code else f"{color.code}  {color.name}"
        self.current_text.configure(text=f"{label}\nRGB {color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]}")

    def select_color(self, index: int) -> None:
        if self.result is None or not 0 <= index < len(self.result.palette):
            return
        self.selected_palette = index
        self._refresh_palette_selection()
        color = self.result.palette[index]
        self.status_var.set(f"已选择 {color.code} · {color.name}；现在可在图纸上按住左键拖动上色。")

    def _mode_changed(self) -> None:
        mode = self.edit_mode_var.get()
        self.canvas.configure(cursor="crosshair" if mode == "paint" else "tcross")
        self.status_var.set("画笔模式：左键拖动上色。" if mode == "paint" else "选择模式：拖动框选，点击行号/列号可选整行/整列。")

    def _left_press(self, event: tk.Event) -> None:
        if self.edit_mode_var.get() == "paint":
            self._begin_stroke(event)
        else:
            self._begin_selection(event)

    def _left_motion(self, event: tk.Event) -> None:
        if self.edit_mode_var.get() == "paint":
            self._continue_stroke(event)
        else:
            self._continue_selection(event)

    def _left_release(self, event: tk.Event) -> None:
        if self.edit_mode_var.get() == "paint":
            self._end_stroke(event)
        else:
            self._continue_selection(event)
            self._selection_status()

    def _canvas_cell(self, event: tk.Event) -> tuple[int, int] | None:
        if self.result is None:
            return None
        px = self.canvas.canvasx(event.x) - self.margin
        py = self.canvas.canvasy(event.y) - self.margin
        x, y = math.floor(px / self.cell), math.floor(py / self.cell)
        if 0 <= x < self.result.width and 0 <= y < self.result.height:
            return x, y
        return None

    def _begin_selection(self, event: tk.Event) -> None:
        if self.result is None:
            return
        px, py = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x = math.floor((px - self.margin) / self.cell)
        y = math.floor((py - self.margin) / self.cell)
        if px < self.margin and 0 <= y < self.result.height:
            self.selection_drag_kind = "rows"
            self.selection_anchor = (0, y)
            self.selection = (0, y, self.result.width - 1, y)
        elif py < self.margin and 0 <= x < self.result.width:
            self.selection_drag_kind = "cols"
            self.selection_anchor = (x, 0)
            self.selection = (x, 0, x, self.result.height - 1)
        else:
            pos = self._canvas_cell(event)
            if pos is None:
                return
            self.selection_drag_kind = "cells"
            self.selection_anchor = pos
            self.selection = (*pos, *pos)
        self._draw_selection()

    def _continue_selection(self, event: tk.Event) -> None:
        if self.result is None or self.selection_anchor is None:
            return
        px, py = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x = max(0, min(self.result.width - 1, math.floor((px - self.margin) / self.cell)))
        y = max(0, min(self.result.height - 1, math.floor((py - self.margin) / self.cell)))
        ax, ay = self.selection_anchor
        if self.selection_drag_kind == "rows":
            self.selection = (0, min(ay, y), self.result.width - 1, max(ay, y))
        elif self.selection_drag_kind == "cols":
            self.selection = (min(ax, x), 0, max(ax, x), self.result.height - 1)
        else:
            self.selection = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
        self._draw_selection()

    def _draw_selection(self) -> None:
        if self.selection_rect is not None:
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
        if self.selection is None:
            return
        x0, y0, x1, y1 = self.selection
        self.selection_rect = self.canvas.create_rectangle(
            self.margin + x0 * self.cell - 1,
            self.margin + y0 * self.cell - 1,
            self.margin + (x1 + 1) * self.cell + 1,
            self.margin + (y1 + 1) * self.cell + 1,
            outline="#1687FF", width=4,
        )
        self.canvas.tag_raise(self.selection_rect)

    def _selection_status(self) -> None:
        if self.selection is None:
            return
        x0, y0, x1, y1 = self.selection
        count = (x1 - x0 + 1) * (y1 - y0 + 1)
        self.status_var.set(f"已选择：第 {x0 + 1}–{x1 + 1} 列，第 {y0 + 1}–{y1 + 1} 行，共 {count} 格。按 Enter 可填色。")

    def select_all(self) -> None:
        if self.result is None:
            return
        self.edit_mode_var.set("select")
        self.selection = (0, 0, self.result.width - 1, self.result.height - 1)
        self.selection_anchor = (0, 0)
        self._draw_selection()
        self._selection_status()

    def select_whole_row(self) -> None:
        if self.result is None:
            return
        y0 = self.selection[1] if self.selection else 0
        y1 = self.selection[3] if self.selection else y0
        self.edit_mode_var.set("select")
        self.selection = (0, y0, self.result.width - 1, y1)
        self._draw_selection()
        self._selection_status()

    def select_whole_column(self) -> None:
        if self.result is None:
            return
        x0 = self.selection[0] if self.selection else 0
        x1 = self.selection[2] if self.selection else x0
        self.edit_mode_var.set("select")
        self.selection = (x0, 0, x1, self.result.height - 1)
        self._draw_selection()
        self._selection_status()

    def fill_selection(self) -> None:
        if self.result is None or self.selection is None:
            self.status_var.set("请先切换到矩形选择模式并选中格子。")
            return
        x0, y0, x1, y1 = self.selection
        changes: list[tuple[int, int, int, int]] = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                old = int(self.result.indices[y, x])
                if old != self.selected_palette:
                    self.result.indices[y, x] = self.selected_palette
                    changes.append((x, y, old, self.selected_palette))
                    self._update_cell(x, y)
        self._commit_changes(changes, "已批量填充")

    def _commit_changes(self, changes: list[tuple[int, int, int, int]], message: str) -> None:
        if not changes:
            self.status_var.set("所选格子已经是当前颜色。")
            return
        self.undo_stack.append(changes)
        self.redo_stack.clear()
        self._mark_dirty()
        self._refresh_summary()
        self._render_palette()
        self._draw_selection()
        self.status_var.set(f"{message} {len(changes)} 格；Ctrl+Z 可撤销。")

    def copy_selection(self) -> None:
        if self.result is None or self.selection is None:
            return
        x0, y0, x1, y1 = self.selection
        rows = [[self.result.palette[int(self.result.indices[y, x])].code for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        self.clipboard_codes = rows
        text = "\n".join("\t".join(row) for row in rows)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set(f"已复制 {(x1 - x0 + 1) * (y1 - y0 + 1)} 格，可粘贴到本软件或 Excel。")

    def paste_selection(self) -> None:
        if self.result is None:
            return
        rows = self.clipboard_codes
        try:
            text = self.root.clipboard_get()
            parsed = [line.split("\t") for line in text.replace("\r", "").split("\n") if line]
            if parsed:
                rows = parsed
        except tk.TclError:
            pass
        if not rows:
            return
        start_x, start_y = (self.selection[0], self.selection[1]) if self.selection else (0, 0)
        code_map = {c.code.upper(): i for i, c in enumerate(self.result.palette)}
        changes: list[tuple[int, int, int, int]] = []
        skipped = 0
        for dy, row in enumerate(rows):
            for dx, code in enumerate(row):
                x, y = start_x + dx, start_y + dy
                if x >= self.result.width or y >= self.result.height:
                    continue
                new = code_map.get(str(code).strip().upper())
                if new is None:
                    skipped += 1
                    continue
                old = int(self.result.indices[y, x])
                if old != new:
                    self.result.indices[y, x] = new
                    changes.append((x, y, old, new))
                    self._update_cell(x, y)
        self._commit_changes(changes, "已粘贴")
        if skipped:
            self.status_var.set(self.status_var.get() + f"；跳过 {skipped} 个未知色号。")

    def _begin_stroke(self, event: tk.Event) -> None:
        self._stroke = {}
        self._paint_cell(self._canvas_cell(event))

    def _continue_stroke(self, event: tk.Event) -> None:
        self._paint_cell(self._canvas_cell(event))

    def _end_stroke(self, _event: tk.Event | None = None) -> None:
        if not self._stroke:
            return
        changes = [(x, y, old, new) for (x, y), (old, new) in self._stroke.items() if old != new]
        self._stroke = {}
        if not changes:
            return
        self.undo_stack.append(changes)
        self.redo_stack.clear()
        self._mark_dirty()
        self._refresh_summary()
        self._render_palette()
        self.status_var.set(f"本次修改 {len(changes)} 格；Ctrl+Z 可撤销。")

    def _paint_cell(self, cell_pos: tuple[int, int] | None) -> None:
        if self.result is None or cell_pos is None:
            return
        x, y = cell_pos
        current = int(self.result.indices[y, x])
        if (x, y) not in self._stroke:
            self._stroke[(x, y)] = (current, self.selected_palette)
        else:
            old, _ = self._stroke[(x, y)]
            self._stroke[(x, y)] = (old, self.selected_palette)
        if current == self.selected_palette:
            return
        self.result.indices[y, x] = self.selected_palette
        self._update_cell(x, y)

    def _pick_color(self, event: tk.Event) -> None:
        pos = self._canvas_cell(event)
        if self.result is None or pos is None:
            return
        x, y = pos
        self.select_color(int(self.result.indices[y, x]))
        color = self.result.palette[self.selected_palette]
        self.status_var.set(f"已从第 {x + 1} 列、第 {y + 1} 行取色：{color.code}")

    def _update_cell(self, x: int, y: int) -> None:
        if self.result is None or not self.cell_rects:
            return
        color = self.result.palette[int(self.result.indices[y, x])]
        self.canvas.itemconfigure(self.cell_rects[y][x], fill=self._hex(color.rgb))
        text_id = self.cell_texts[y][x]
        if text_id is not None:
            self.canvas.itemconfigure(text_id, text=color.code, fill=self._text_hex(color.rgb))

    def undo(self) -> None:
        if self.result is None or not self.undo_stack:
            self.status_var.set("没有可撤销的修改。")
            return
        changes = self.undo_stack.pop()
        for x, y, old, _new in changes:
            self.result.indices[y, x] = old
            self._update_cell(x, y)
        self.redo_stack.append(changes)
        self._mark_dirty()
        self._refresh_summary()
        self._render_palette()
        self.status_var.set(f"已撤销 {len(changes)} 格。")

    def redo(self) -> None:
        if self.result is None or not self.redo_stack:
            self.status_var.set("没有可重做的修改。")
            return
        changes = self.redo_stack.pop()
        for x, y, _old, new in changes:
            self.result.indices[y, x] = new
            self._update_cell(x, y)
        self.undo_stack.append(changes)
        self._mark_dirty()
        self._refresh_summary()
        self._render_palette()
        self.status_var.set(f"已重做 {len(changes)} 格。")

    def change_zoom(self, direction: int) -> None:
        new_index = max(0, min(len(self.ZOOMS) - 1, self.zoom_index + direction))
        if new_index == self.zoom_index:
            return
        self.zoom_index = new_index
        self._render_grid()

    def fit_view(self) -> None:
        if self.result is None:
            return
        available_w = max(300, self.canvas.winfo_width() - self.margin * 2 - 10)
        available_h = max(300, self.canvas.winfo_height() - self.margin * 2 - 10)
        target = min(available_w / self.result.width, available_h / self.result.height)
        self.zoom_index = min(range(len(self.ZOOMS)), key=lambda i: abs(self.ZOOMS[i] - target))
        self._render_grid(keep_center=(0, 0))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _mousewheel(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _shift_mousewheel(self, event: tk.Event) -> str:
        self.canvas.xview_scroll(-int(event.delta / 120), "units")
        return "break"

    def _refresh_summary(self) -> None:
        if self.result is None:
            return
        used = len(self.result.counts())
        boards_x = math.ceil(self.result.width / 29)
        boards_y = math.ceil(self.result.height / 29)
        self.summary_label.configure(
            text=(
                f"尺寸：{self.result.width} × {self.result.height}\n"
                f"总豆数：{self.result.width * self.result.height}\n"
                f"当前用色：{used} 种\n"
                f"29×29 底板：{boards_x} × {boards_y} 块"
            )
        )

    def open_project(self) -> bool:
        if not self._confirm_replace():
            return False
        path = filedialog.askopenfilename(
            title="打开 MOSAIBeads 项目",
            filetypes=[("MOSAIBeads 项目", "*.json"), ("所有文件", "*.*")],
            parent=self.root,
        )
        if not path:
            return False
        try:
            loaded = load_project(path)
        except Exception as exc:
            messagebox.showerror("项目无法打开", str(exc), parent=self.root)
            return False
        self._install_project(loaded)
        self.current_path = Path(path)
        self.version_var.set("外部项目")
        self._set_title(self.current_path.stem)
        return True

    def save_current(self) -> bool:
        if self.result is None or self.loaded is None:
            return False
        initial = self.current_path.name if self.current_path else f"{self._project_name().replace('（58×57）', '')}_手工调整.json"
        path = filedialog.asksaveasfilename(
            title="保存可继续编辑的项目",
            defaultextension=".json",
            initialfile=initial,
            filetypes=[("MOSAIBeads 项目", "*.json")],
            parent=self.root,
        )
        if not path:
            return False
        try:
            save_project(path, self.result, self.loaded.raw)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return False
        self.current_path = Path(path)
        self.loaded.raw = project_dict(self.result, self.loaded.raw)
        self.loaded.source_path = self.current_path
        self.dirty = False
        self._set_title(self.current_path.stem)
        self.status_var.set(f"项目已保存：{self.current_path}")
        return True

    def export_current(self) -> None:
        if self.result is None:
            return
        folder = filedialog.askdirectory(title="选择图纸导出文件夹", parent=self.root)
        if not folder:
            return
        safe_name = self.current_path.stem if self.current_path else "couple_58x57_manual"
        output_dir = Path(folder) / f"{safe_name}_图纸"
        self.status_var.set("正在导出 PNG、PDF、材料表、项目文件和 Excel……")
        self.root.update_idletasks()
        try:
            outputs = export_bundle(self.result, output_dir, safe_name)
            save_project(output_dir / f"{safe_name}_project.json", self.result, self.loaded.raw if self.loaded else None)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            self.status_var.set("导出失败。")
            return
        self.status_var.set(f"已导出 {len(outputs)} 个文件：{output_dir}")
        if messagebox.askyesno("导出完成", f"图纸已导出到：\n{output_dir}\n\n是否打开文件夹？", parent=self.root):
            os.startfile(output_dir)

    def on_close(self) -> None:
        if self._confirm_replace():
            self.root.destroy()

    @staticmethod
    def _hex(rgb: tuple[int, int, int]) -> str:
        return "#%02X%02X%02X" % rgb

    @staticmethod
    def _text_hex(rgb: tuple[int, int, int]) -> str:
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return "#151922" if lum > 150 else "#FFFFFF"


def run() -> None:
    root = tk.Tk()
    ManualEditor(root)
    root.mainloop()


def smoke_test() -> None:
    for label in BUILTIN_PROJECTS:
        loaded = load_project(builtin_project_path(label))
        result = loaded.result
        if (result.width, result.height) != (58, 57):
            raise RuntimeError(f"{label} 的尺寸不是 58×57")
        if sum(count for _color, count in result.counts()) != 3306:
            raise RuntimeError(f"{label} 的豆数无效")
        if len(result.palette) < 291:
            raise RuntimeError(f"{label} 没有加载完整的 MARD 291 品牌色板")
