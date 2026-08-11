from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from .engine import ConvertOptions, PatternResult, convert_image, recommend_bead_colors
from .palettes import BeadColor


class FixedPaletteDemo(tk.Toplevel):
    """Click-to-sample dialog whose output is guaranteed to use an exact allow-list."""

    def __init__(self, parent: tk.Misc, image: Image.Image, palette: list[BeadColor],
                 options: ConvertOptions, on_result):
        super().__init__(parent)
        self.title("指定豆色 Demo · 点击原图取色")
        self.geometry("1020x700")
        self.minsize(850, 580)
        self.image = image.convert("RGB")
        self.palette = palette
        self.options = options
        self.on_result = on_result
        self.selected: list[BeadColor] = []
        self.recommendations: list[tuple[BeadColor, float]] = []
        self.photo: ImageTk.PhotoImage | None = None
        self.view = (0.0, 0.0, 1.0)
        self._build()
        self.after(50, self._draw_image)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="点击原图中的颜色区域，确认推荐豆色；最终图纸严格只用已选颜色",
                  style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, width=330, padding=(14, 0, 0, 0))
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.canvas = tk.Canvas(left, bg="#101216", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Configure>", lambda _e: self._draw_image())

        ttk.Label(right, text="相近的真实豆色", style="Section.TLabel").pack(anchor="w")
        self.hint = tk.StringVar(value="请点击左侧图片")
        ttk.Label(right, textvariable=self.hint, style="Muted.TLabel", wraplength=305).pack(anchor="w", pady=(3, 7))
        self.candidate_frame = ttk.Frame(right)
        self.candidate_frame.pack(fill="x")
        ttk.Separator(right).pack(fill="x", pady=12)
        ttk.Label(right, text="已选颜色（精确限制）", style="Section.TLabel").pack(anchor="w")
        self.listbox = tk.Listbox(right, bg="#2a2e36", fg="#f2f3f5", selectbackground="#d77f35",
                                  selectforeground="white", relief="flat", font=("Microsoft YaHei UI", 10),
                                  height=12)
        self.listbox.pack(fill="both", expand=True, pady=6)
        row = ttk.Frame(right)
        row.pack(fill="x")
        ttk.Button(row, text="移除选中", command=self._remove).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="全部清空", command=self._clear).pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.progress = ttk.Progressbar(right, maximum=100)
        self.progress.pack(fill="x", pady=(12, 5))
        self.status = tk.StringVar(value="建议选择 4–16 种颜色")
        ttk.Label(right, textvariable=self.status, style="Muted.TLabel", wraplength=305).pack(anchor="w")
        self.generate = ttk.Button(right, text="只用这些颜色生成", style="Accent.TButton", command=self._generate)
        self.generate.pack(fill="x", pady=(8, 0))

    def _draw_image(self) -> None:
        if not self.winfo_exists():
            return
        self.canvas.delete("image")
        cw, ch = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        scale = min((cw - 20) / self.image.width, (ch - 20) / self.image.height)
        size = (max(1, round(self.image.width * scale)), max(1, round(self.image.height * scale)))
        self.photo = ImageTk.PhotoImage(self.image.resize(size, Image.Resampling.LANCZOS))
        x, y = (cw - size[0]) / 2, (ch - size[1]) / 2
        self.canvas.create_image(x, y, image=self.photo, anchor="nw", tags="image")
        self.canvas.tag_lower("image")
        self.view = (x, y, scale)

    def _click(self, event) -> None:
        ox, oy, scale = self.view
        x, y = round((event.x - ox) / scale), round((event.y - oy) / scale)
        if not (0 <= x < self.image.width and 0 <= y < self.image.height):
            return
        self.canvas.delete("marker")
        self.canvas.create_oval(event.x-7, event.y-7, event.x+7, event.y+7,
                                outline="#ffb76d", width=3, tags="marker")
        radius = max(1, round(max(self.image.size) / 450))
        self.recommendations = recommend_bead_colors(self.image, x, y, self.palette, radius, 6)
        self.hint.set(f"原图坐标 ({x}, {y})；点击一种推荐色加入下方")
        self._show_candidates()

    def _show_candidates(self) -> None:
        for child in self.candidate_frame.winfo_children():
            child.destroy()
        for color, distance in self.recommendations:
            row = ttk.Frame(self.candidate_frame)
            row.pack(fill="x", pady=2)
            swatch = tk.Canvas(row, width=28, height=25, highlightthickness=1,
                               highlightbackground="#626873", bg="#%02x%02x%02x" % color.rgb)
            swatch.pack(side="left", padx=(0, 7))
            label = color.code if color.name == color.code else f"{color.code}  {color.name}"
            ttk.Button(row, text=f"{label}   ΔE {distance:.1f}",
                       command=lambda c=color: self._add(c)).pack(side="left", fill="x", expand=True)

    def _add(self, color: BeadColor) -> None:
        if color not in self.selected:
            self.selected.append(color)
            self._refresh_selected()

    def _refresh_selected(self) -> None:
        self.listbox.delete(0, "end")
        for color in self.selected:
            self.listbox.insert("end", f"  {color.code}   {color.name}   RGB {color.rgb}")
        self.status.set(f"已选 {len(self.selected)} 种；生成结果不会出现其他颜色")

    def _remove(self) -> None:
        choice = self.listbox.curselection()
        if choice:
            self.selected.pop(int(choice[0]))
            self._refresh_selected()

    def _clear(self) -> None:
        self.selected.clear()
        self._refresh_selected()

    def _generate(self) -> None:
        if len(self.selected) < 2:
            messagebox.showinfo("颜色不足", "请从原图至少确认 2 种豆色。", parent=self)
            return
        self.generate.state(["disabled"])
        self.progress["value"] = 0
        self.status.set("正在使用固定色板生成…")
        colors = list(self.selected)
        options = replace(self.options, fixed_palette=True, max_colors=len(colors))

        def progress(value: int, text: str) -> None:
            self.after(0, lambda: (self.progress.configure(value=value), self.status.set(text)))

        def work() -> None:
            try:
                result = convert_image(self.image, colors, options, progress)
                self.after(0, lambda: self._finish(result))
            except Exception as exc:
                self.after(0, lambda: self._fail(str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _finish(self, result: PatternResult) -> None:
        self.on_result(result)
        self.destroy()

    def _fail(self, detail: str) -> None:
        self.generate.state(["!disabled"])
        self.status.set("生成失败")
        messagebox.showerror("固定色板生成失败", detail, parent=self)
