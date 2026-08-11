from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .ai_model import inspect_ai_model, load_semantic_backend
from .autotune import (SCHEME_BALANCED, SCHEME_CRAFT, SCHEME_LIKENESS,
                       AutoTuneBundle, auto_tune_all)
from .engine import ConvertOptions, PatternResult, convert_image
from .exporter import export_bundle, render_clean
from .palettes import available_palettes, load_palette
from .palette_demo import FixedPaletteDemo


BG = "#17191e"
PANEL = "#22252c"
PANEL_2 = "#2a2e36"
TEXT = "#f2f3f5"
MUTED = "#a9afb9"
ACCENT = "#f3a65a"
ACCENT_DARK = "#d77f35"
TUNABLE_KEYS = ("palette", "background", "width", "max_colors", "profile",
                "detail", "cleanup", "saturation", "contrast", "dither")


class BeadSketchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MOSAIBEADS 3.0 · 智能拼豆图纸")
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w, win_h = min(1440, screen_w - 80), min(900, screen_h - 100)
        self.geometry(f"{win_w}x{win_h}+{max(0, (screen_w-win_w)//2)}+{max(0, (screen_h-win_h)//2)}")
        self.minsize(1220, 760)
        self.configure(bg=BG)
        self.source_image: Image.Image | None = None
        self.source_path: Path | None = None
        self.result: PatternResult | None = None
        self.auto_bundle: AutoTuneBundle | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.source_photo: ImageTk.PhotoImage | None = None
        self.display_info = (0.0, 0.0, 1.0)
        self.selected_color = 0
        self.swatch_buttons: list[tk.Button] = []
        self.undo_stack: list[tuple[int, int, int, int]] = []
        self.redo_stack: list[tuple[int, int, int, int]] = []
        self.work_queue: queue.Queue = queue.Queue()
        self._poll_after_id: str | None = None
        self._build_style()
        self._build_ui()
        self._poll_after_id = self.after(80, self._poll_queue)
        self.bind("<Control-o>", lambda _e: self.open_image())
        self.bind("<Control-e>", lambda _e: self.export())
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-y>", lambda _e: self.redo())

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, fieldbackground=PANEL_2,
                        bordercolor="#3b404a", lightcolor="#3b404a", darkcolor="#3b404a")
        style.configure("TFrame", background=PANEL)
        style.configure("Dark.TFrame", background=BG)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 17, "bold"), foreground="#fff8ef")
        style.configure("Section.TLabel", font=("Microsoft YaHei UI", 11, "bold"), foreground="#ffd2a4")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(11, 8))
        style.map("TButton", background=[("active", "#3a3f49")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#24180d", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#ffc07f"), ("disabled", "#6d6258")])
        # Windows/clam overrides the foreground for readonly comboboxes. A dedicated
        # style and explicit state maps keep selected values readable in every state.
        style.configure("Readable.TCombobox", padding=5, arrowsize=16,
                        foreground="#f6f7f9", fieldbackground="#343943",
                        background="#343943", arrowcolor=ACCENT,
                        bordercolor="#59606d", lightcolor="#59606d", darkcolor="#59606d",
                        selectbackground="#343943", selectforeground="#f6f7f9")
        style.map("Readable.TCombobox",
                  foreground=[("disabled", "#969daa"), ("readonly", "#f6f7f9")],
                  fieldbackground=[("disabled", "#2b2f37"), ("readonly", "#343943")],
                  background=[("active", "#424854"), ("readonly", "#343943")],
                  arrowcolor=[("disabled", "#777e89"), ("readonly", ACCENT)],
                  selectbackground=[("disabled", "#2b2f37"), ("readonly", "#343943")],
                  selectforeground=[("disabled", "#969daa"), ("readonly", "#f6f7f9")])
        self.option_add("*TCombobox*Listbox.background", "#343943")
        self.option_add("*TCombobox*Listbox.foreground", "#f6f7f9")
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT_DARK)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure("Horizontal.TProgressbar", troughcolor="#323640", background=ACCENT, bordercolor="#323640")
        style.configure("Treeview", background=PANEL_2, fieldbackground=PANEL_2, foreground=TEXT, rowheight=28,
                        borderwidth=0, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", background="#343943", foreground="#e9ebef", font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT_DARK)], foreground=[("selected", "white")])

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Dark.TFrame", padding=(18, 10))
        header.pack(fill="x")
        brand = ttk.Frame(header, style="Dark.TFrame")
        brand.pack(side="left", fill="y", padx=(0, 14))
        ttk.Label(brand, text="MOSAIBEADS 3.0", style="Title.TLabel", background=BG).pack(anchor="nw", pady=(5, 0))

        actions = ttk.Frame(header, style="Dark.TFrame")
        actions.pack(side="right", fill="y", padx=(12, 0))
        ttk.Button(actions, text="导出图纸  Ctrl+E", style="Accent.TButton", command=self.export).pack(anchor="e", fill="x", pady=(5, 6))
        ttk.Button(actions, text="打开图片  Ctrl+O", command=self.open_image).pack(anchor="e", fill="x")

        self.top_controls = ttk.Frame(header, style="Dark.TFrame")
        self.top_controls.pack(side="left", fill="both", expand=True)

        body = ttk.Frame(self, style="Dark.TFrame")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        left_shell = ttk.Frame(body, width=306)
        left_shell.pack(side="left", fill="y", padx=(0, 10))
        left_shell.pack_propagate(False)
        self.left_canvas = tk.Canvas(left_shell, bg=PANEL, highlightthickness=0, width=284,
                                     yscrollincrement=12)
        left_scroll = ttk.Scrollbar(left_shell, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left = ttk.Frame(self.left_canvas, padding=15)
        left_window = self.left_canvas.create_window((0, 0), window=self.left, anchor="nw")
        self.left.bind("<Configure>", lambda _e: self.left_canvas.configure(
            scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas.bind("<Configure>", lambda e: self.left_canvas.itemconfigure(
            left_window, width=e.width))
        # Bind at application level, then act only while the pointer is inside the
        # left column. This is how a web page behaves: controls under the pointer do
        # not swallow wheel/touchpad scrolling.
        self.bind_all("<MouseWheel>", self._scroll_left_web_style, add="+")
        self.bind_all("<Button-4>", lambda e: self._scroll_left_linux(e, -1), add="+")
        self.bind_all("<Button-5>", lambda e: self._scroll_left_linux(e, 1), add="+")
        self.right = ttk.Frame(body, width=330, padding=12)
        self.right.pack(side="right", fill="y", padx=(10, 0))
        self.right.pack_propagate(False)
        # Reserve the fixed-width sidebars before the expanding center. Packing the
        # center first lets it consume the right palette's space on 1360 px screens.
        self.center = ttk.Frame(body, style="Dark.TFrame")
        self.center.pack(side="left", fill="both", expand=True)
        self._build_controls()
        self._build_canvases()
        self._build_palette_panel()

    def _section(self, text: str) -> None:
        ttk.Label(self.left, text=text, style="Section.TLabel").pack(anchor="w", pady=(12, 6))

    def _parameter_label(self, label: str, lock_key: str, pady=(0, 0)) -> None:
        row = ttk.Frame(self.left)
        row.pack(fill="x", pady=pady)
        ttk.Label(row, text=label, style="Muted.TLabel").pack(side="left")
        ttk.Checkbutton(row, text="固定", variable=self.lock_vars[lock_key]).pack(side="right")

    def _scale(self, label: str, variable: tk.Variable, start: float, end: float,
               resolution: float, lock_key: str) -> None:
        row = ttk.Frame(self.left)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Checkbutton(row, text="固定", variable=self.lock_vars[lock_key]).pack(side="right")
        ttk.Label(row, textvariable=variable, style="Muted.TLabel").pack(side="right", padx=(0, 7))
        tk.Scale(self.left, variable=variable, from_=start, to=end, resolution=resolution,
                 orient="horizontal", showvalue=False, bg=PANEL, fg=TEXT, highlightthickness=0,
                 troughcolor="#343943", activebackground=ACCENT, sliderrelief="flat").pack(fill="x")

    def _build_controls(self) -> None:
        self.palette_var = tk.StringVar(value="MARD 291")
        self.background_var = tk.StringVar(value="白色")
        self.profile_var = tk.StringVar(value="自动")
        self.width_var = tk.IntVar(value=48)
        self.colors_var = tk.IntVar(value=24)
        self.detail_var = tk.DoubleVar(value=0.72)
        self.cleanup_var = tk.DoubleVar(value=0.54)
        self.saturation_var = tk.DoubleVar(value=1.06)
        self.contrast_var = tk.DoubleVar(value=1.05)
        self.dither_var = tk.StringVar(value="关闭")
        self.holes_var = tk.BooleanVar(value=True)
        self.grid_var = tk.BooleanVar(value=True)
        self.ai_var = tk.BooleanVar(value=True)
        self.scheme_var = tk.StringVar(value=SCHEME_BALANCED)
        self.lock_vars = {key: tk.BooleanVar(value=False) for key in TUNABLE_KEYS}
        self._section("1 · 尺寸与色板")
        self._parameter_label("品牌色板", "palette")
        ttk.Combobox(self.left, textvariable=self.palette_var, values=available_palettes(),
                     state="readonly", style="Readable.TCombobox").pack(fill="x", pady=(3, 5))
        self._parameter_label("透明区域背景", "background")
        ttk.Combobox(self.left, textvariable=self.background_var,
                     values=["白色", "黑色", "透明区域留白"], state="readonly",
                     style="Readable.TCombobox").pack(fill="x", pady=(3, 5))
        self._scale("横向豆数", self.width_var, 16, 120, 1, "width")
        self._scale("最多颜色", self.colors_var, 4, 48, 1, "max_colors")
        ttk.Label(self.left, text="29 格 = 1 块常见大板；宽度 48 适合头像", style="Muted.TLabel", wraplength=245).pack(anchor="w", pady=3)

        self._section("2 · 传神程度")
        self._parameter_label("图片类型", "profile")
        ttk.Combobox(self.left, textvariable=self.profile_var,
                     values=["自动", "人像/宠物", "插画/动漫", "照片"], state="readonly",
                     style="Readable.TCombobox").pack(fill="x", pady=(3, 5))
        self._scale("特征保护", self.detail_var, 0.0, 1.0, 0.01, "detail")
        self._scale("杂色清理", self.cleanup_var, 0.0, 1.0, 0.01, "cleanup")
        self._scale("饱和度", self.saturation_var, 0.7, 1.4, 0.01, "saturation")
        self._scale("对比度", self.contrast_var, 0.75, 1.35, 0.01, "contrast")
        self._parameter_label("抖动", "dither", pady=(5, 0))
        ttk.Combobox(self.left, textvariable=self.dither_var, values=["关闭", "轻微", "明显"],
                     state="readonly", style="Readable.TCombobox").pack(fill="x", pady=3)
        ttk.Label(self.left, text="人像通常关闭抖动；风景渐变可选“轻微”", style="Muted.TLabel", wraplength=245).pack(anchor="w")

        self._build_top_controls()

    def _build_top_controls(self) -> None:
        smart_card = ttk.LabelFrame(self.top_controls, text="3 · 智能方案", padding=(10, 6))
        smart_card.pack(side="left", fill="both", expand=True, padx=(0, 7))
        self.ai_status = inspect_ai_model()

        smart_meta = ttk.Frame(smart_card)
        smart_meta.pack(fill="x")
        ttk.Checkbutton(smart_meta, text="AI 语义评分", variable=self.ai_var).pack(side="left")
        ttk.Label(smart_meta, text=self.ai_status.provider, style="Muted.TLabel").pack(side="right")

        smart_actions = ttk.Frame(smart_card)
        smart_actions.pack(fill="x", pady=(4, 3))
        self.smart_btn = ttk.Button(smart_actions, text="智能调参（3 套）",
                                    style="Accent.TButton", command=self.smart_tune)
        self.smart_btn.pack(side="left", fill="x", expand=True)
        self.scheme_combo = ttk.Combobox(smart_actions, textvariable=self.scheme_var,
                                         values=[SCHEME_LIKENESS, SCHEME_BALANCED, SCHEME_CRAFT],
                                         state="readonly", width=11, style="Readable.TCombobox")
        self.scheme_combo.pack(side="left", padx=(6, 0))
        self.scheme_combo.bind("<<ComboboxSelected>>", self._switch_scheme)

        smart_bottom = ttk.Frame(smart_card)
        smart_bottom.pack(fill="x")
        ttk.Label(smart_bottom, text="固定项不参与搜索", style="Muted.TLabel").pack(side="left")
        ttk.Button(smart_bottom, text="指定豆色", command=self.open_palette_demo).pack(side="right")

        refine_card = ttk.LabelFrame(self.top_controls, text="4 · 预览与精修", padding=(10, 6))
        refine_card.pack(side="left", fill="both", expand=True, padx=(0, 2))

        refine_top = ttk.Frame(refine_card)
        refine_top.pack(fill="x")
        ttk.Checkbutton(refine_top, text="豆孔", variable=self.holes_var,
                        command=self.refresh_pattern).pack(side="left")
        ttk.Checkbutton(refine_top, text="网格", variable=self.grid_var,
                        command=self.refresh_pattern).pack(side="left", padx=(6, 0))
        ttk.Button(refine_top, text="撤销", command=self.undo).pack(side="right")
        ttk.Button(refine_top, text="重做", command=self.redo).pack(side="right", padx=(0, 5))

        refine_actions = ttk.Frame(refine_card)
        refine_actions.pack(fill="x", pady=(4, 3))
        self.generate_btn = ttk.Button(refine_actions, text="按当前参数生成", command=self.generate)
        self.generate_btn.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(refine_actions, mode="determinate", maximum=100, length=120)
        self.progress.pack(side="left", fill="x", expand=True, padx=(7, 0))

        self.status_var = tk.StringVar(value="请先打开一张图片")
        ttk.Label(refine_card, textvariable=self.status_var, style="Muted.TLabel",
                  wraplength=355).pack(anchor="w")

    def _pointer_in_left_column(self) -> bool:
        x, y = self.winfo_pointerxy()
        left, top = self.left_canvas.winfo_rootx(), self.left_canvas.winfo_rooty()
        return left <= x < left + self.left_canvas.winfo_width() and top <= y < top + self.left_canvas.winfo_height()

    def _scroll_left_web_style(self, event):
        if not self._pointer_in_left_column():
            return None
        delta = int(getattr(event, "delta", 0))
        if not delta:
            return "break"
        # Conventional wheels report multiples of 120; precision touchpads often
        # report smaller deltas. Preserve both and keep the motion comfortably quick.
        steps = -round(delta / 40) if abs(delta) >= 40 else (-1 if delta > 0 else 1)
        self.left_canvas.yview_scroll(steps, "units")
        return "break"

    def _scroll_left_linux(self, _event, direction: int):
        if self._pointer_in_left_column():
            self.left_canvas.yview_scroll(direction * 3, "units")
            return "break"
        return None

    def _build_canvases(self) -> None:
        panes = ttk.Panedwindow(self.center, orient="horizontal")
        panes.pack(fill="both", expand=True)
        source_frame = ttk.Frame(panes, padding=8)
        result_frame = ttk.Frame(panes, padding=8)
        panes.add(source_frame, weight=1)
        panes.add(result_frame, weight=1)
        ttk.Label(source_frame, text="原图", style="Section.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(result_frame, text="拼豆效果 · 右侧选色后点击或拖动格子填色",
                  style="Section.TLabel").pack(anchor="w", pady=(0, 5))
        self.source_canvas = tk.Canvas(source_frame, bg="#101216", highlightthickness=0)
        self.source_canvas.pack(fill="both", expand=True)
        self.pattern_canvas = tk.Canvas(result_frame, bg="#101216", highlightthickness=0, cursor="crosshair")
        self.pattern_canvas.pack(fill="both", expand=True)
        self.pattern_canvas.bind("<Button-1>", self._paint_event)
        self.pattern_canvas.bind("<B1-Motion>", self._paint_event)
        self.source_canvas.bind("<Configure>", lambda _e: self.refresh_source())
        self.pattern_canvas.bind("<Configure>", lambda _e: self.refresh_pattern())

    def _build_palette_panel(self) -> None:
        ttk.Label(self.right, text="拼豆色号调色板", style="Section.TLabel").pack(anchor="w", pady=(0, 3))
        ttk.Label(self.right, text="点选色块，再点击或拖动图纸格子填色",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 7))

        selected = ttk.Frame(self.right)
        selected.pack(fill="x", pady=(0, 7))
        self.selected_chip = tk.Canvas(selected, width=30, height=30, bg=PANEL,
                                       highlightthickness=2, highlightbackground=ACCENT)
        self.selected_chip.pack(side="left")
        self.selected_color_var = tk.StringVar(value="尚未选择颜色")
        ttk.Label(selected, textvariable=self.selected_color_var).pack(side="left", padx=(8, 0))

        swatch_shell = ttk.Frame(self.right)
        swatch_shell.pack(fill="x", pady=(0, 9))
        self.swatch_canvas = tk.Canvas(swatch_shell, height=210, bg=PANEL_2,
                                       highlightthickness=1, highlightbackground="#3b404a",
                                       yscrollincrement=20)
        swatch_scroll = ttk.Scrollbar(swatch_shell, orient="vertical", command=self.swatch_canvas.yview)
        self.swatch_canvas.configure(yscrollcommand=swatch_scroll.set)
        swatch_scroll.pack(side="right", fill="y")
        self.swatch_canvas.pack(side="left", fill="x", expand=True)
        self.swatch_grid = ttk.Frame(self.swatch_canvas)
        swatch_window = self.swatch_canvas.create_window((0, 0), window=self.swatch_grid, anchor="nw")
        self.swatch_grid.bind("<Configure>", lambda _e: self.swatch_canvas.configure(
            scrollregion=self.swatch_canvas.bbox("all")))
        self.swatch_canvas.bind("<Configure>", lambda e: self.swatch_canvas.itemconfigure(
            swatch_window, width=e.width))
        self.swatch_canvas.bind("<MouseWheel>", self._scroll_swatches)

        ttk.Label(self.right, text="色号与用量", style="Section.TLabel").pack(anchor="w", pady=(0, 5))
        self.color_tree = ttk.Treeview(self.right, columns=("code", "count"), show="headings", selectmode="browse")
        self.color_tree.heading("code", text="色号 / 名称")
        self.color_tree.heading("count", text="数量")
        self.color_tree.column("code", width=175, anchor="w")
        self.color_tree.column("count", width=62, anchor="e")
        self.color_tree.pack(fill="both", expand=True)
        self.color_tree.bind("<<TreeviewSelect>>", self._select_tree_color)
        info = ttk.Frame(self.right)
        info.pack(fill="x", pady=(10, 0))
        self.info_var = tk.StringVar(value="尚未生成图纸")
        ttk.Label(info, textvariable=self.info_var, style="Muted.TLabel", wraplength=295).pack(anchor="w")

    def _scroll_swatches(self, event):
        delta = int(getattr(event, "delta", 0))
        if delta:
            steps = -round(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            self.swatch_canvas.yview_scroll(steps, "units")
        return "break"

    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.source_image = im.copy()
            self.source_path = Path(path)
            self.result = None
            self.auto_bundle = None
            self.status_var.set(f"已打开：{self.source_path.name}")
            self.refresh_source()
            self.refresh_pattern()
            self.after(100, self.generate)
        except Exception as exc:
            messagebox.showerror("无法打开", str(exc))

    def _fit_photo(self, image: Image.Image, canvas: tk.Canvas, pixelated: bool = False) -> tuple[ImageTk.PhotoImage, float, float, float]:
        cw, ch = max(40, canvas.winfo_width()), max(40, canvas.winfo_height())
        scale = min((cw - 22) / image.width, (ch - 22) / image.height)
        nw, nh = max(1, int(image.width * scale)), max(1, int(image.height * scale))
        resized = image.resize((nw, nh), Image.Resampling.NEAREST if pixelated else Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized), (cw - nw) / 2, (ch - nh) / 2, scale

    def refresh_source(self) -> None:
        self.source_canvas.delete("all")
        if not self.source_image:
            self.source_canvas.create_text(self.source_canvas.winfo_width()/2, self.source_canvas.winfo_height()/2,
                                           text="打开任意照片、插画或头像\n支持 PNG / JPG / WebP", fill=MUTED,
                                           font=("Microsoft YaHei UI", 13), justify="center")
            return
        image = self.source_image.convert("RGB")
        self.source_photo, x, y, _ = self._fit_photo(image, self.source_canvas)
        self.source_canvas.create_image(x, y, image=self.source_photo, anchor="nw")

    def _preview_image(self) -> Image.Image:
        assert self.result is not None
        scale = 24
        image = render_clean(self.result, scale=scale, bead_holes=self.holes_var.get())
        if self.grid_var.get():
            draw = ImageDraw.Draw(image)
            minor = (35, 37, 42)
            for x in range(self.result.width + 1):
                width = 3 if x % 29 == 0 else (2 if x % 5 == 0 else 1)
                draw.line((x*scale, 0, x*scale, image.height), fill=minor, width=width)
            for y in range(self.result.height + 1):
                width = 3 if y % 29 == 0 else (2 if y % 5 == 0 else 1)
                draw.line((0, y*scale, image.width, y*scale), fill=minor, width=width)
        return image

    def refresh_pattern(self) -> None:
        self.pattern_canvas.delete("all")
        if not self.result:
            self.pattern_canvas.create_text(self.pattern_canvas.winfo_width()/2, self.pattern_canvas.winfo_height()/2,
                                            text="智能图纸会显示在这里", fill=MUTED, font=("Microsoft YaHei UI", 13))
            return
        self.preview_photo, x, y, scale = self._fit_photo(self._preview_image(), self.pattern_canvas, pixelated=True)
        self.pattern_canvas.create_image(x, y, image=self.preview_photo, anchor="nw")
        self.display_info = (x, y, scale * 24)

    def _convert_options(self) -> ConvertOptions:
        return ConvertOptions(width=self.width_var.get(), max_colors=self.colors_var.get(),
                              profile=self.profile_var.get(), detail=self.detail_var.get(),
                              cleanup=self.cleanup_var.get(), saturation=self.saturation_var.get(),
                              contrast=self.contrast_var.get(), dither=self.dither_var.get(),
                              background=self.background_var.get())

    def generate(self) -> None:
        if not self.source_image:
            self.open_image()
            return
        self.generate_btn.state(["disabled"])
        self.progress["value"] = 0
        self.status_var.set("准备生成…")
        image, opts, palette_name = self.source_image.copy(), self._convert_options(), self.palette_var.get()

        def worker() -> None:
            try:
                palette = load_palette(palette_name)
                result = convert_image(image, palette, opts,
                                       lambda n, msg: self.work_queue.put(("progress", n, msg)))
                self.work_queue.put(("done", result))
            except Exception:
                self.work_queue.put(("error", traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def smart_tune(self) -> None:
        if not self.source_image:
            self.open_image()
            return
        locked = {key for key, variable in self.lock_vars.items() if variable.get()}
        if locked == set(TUNABLE_KEYS):
            messagebox.showinfo("没有可调参数", "所有参数都已固定。请至少取消一个“固定”复选框。")
            return
        self.generate_btn.state(["disabled"])
        self.smart_btn.state(["disabled"])
        self.progress["value"] = 0
        self.status_var.set("准备智能搜索…")
        image, opts, palette_name = self.source_image.copy(), self._convert_options(), self.palette_var.get()
        use_ai = bool(self.ai_var.get())

        def worker() -> None:
            try:
                palettes = {name: load_palette(name) for name in available_palettes()}
                backend = load_semantic_backend(prefer_gpu=True) if use_ai else None
                bundle = auto_tune_all(image, palettes, opts, palette_name, backend,
                                       lambda n, msg: self.work_queue.put(("progress", n, msg)),
                                       locked=locked)
                self.work_queue.put(("autotune_done", bundle))
            except Exception:
                self.work_queue.put(("error", traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def _apply_result(self, result: PatternResult) -> None:
        self.result = result
        self.selected_color = 0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._rebuild_swatch_palette()
        self._fill_color_tree()
        self.refresh_pattern()
        self._refresh_info()

    def _switch_scheme(self, _event=None) -> None:
        if self.auto_bundle and self.scheme_var.get() in self.auto_bundle.results:
            scheme = self.scheme_var.get()
            result = self.auto_bundle.results[scheme]
            opts = self.auto_bundle.options[scheme]
            palette_name = self.auto_bundle.palette_names.get(scheme, self.palette_var.get())
            self.palette_var.set(palette_name)
            self.background_var.set(opts.background)
            self.width_var.set(opts.width)
            self.colors_var.set(opts.max_colors)
            self.profile_var.set(opts.profile)
            self.detail_var.set(round(opts.detail, 2))
            self.cleanup_var.set(round(opts.cleanup, 2))
            self.saturation_var.set(round(opts.saturation, 2))
            self.contrast_var.set(round(opts.contrast, 2))
            self.dither_var.set(opts.dither)
            self._apply_result(result)
            score = self.auto_bundle.scores[scheme]
            self.status_var.set(
                f"{scheme}：{palette_name} · {opts.width} 格 · 最多 {opts.max_colors} 色 · "
                f"传神 {score.likeness*100:.1f}")

    def open_palette_demo(self) -> None:
        if not self.source_image:
            self.open_image()
            return
        palette = load_palette(self.palette_var.get())
        FixedPaletteDemo(self, self.source_image.copy(), palette, self._convert_options(),
                         self._accept_demo_result)

    def _accept_demo_result(self, result: PatternResult) -> None:
        self.auto_bundle = None
        self._apply_result(result)
        self.status_var.set(f"固定色板完成：严格使用所选 {len(result.palette)} 种豆色")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.work_queue.get_nowait()
                if item[0] == "progress":
                    self.progress["value"], msg = item[1], item[2]
                    self.status_var.set(msg)
                elif item[0] == "done":
                    self.auto_bundle = None
                    self._apply_result(item[1])
                    self.generate_btn.state(["!disabled"])
                    self.smart_btn.state(["!disabled"])
                    self.status_var.set(f"完成：{self.result.width} × {self.result.height}，{len(self.result.counts())} 色")
                elif item[0] == "autotune_done":
                    self.auto_bundle = item[1]
                    self.generate_btn.state(["!disabled"])
                    self.smart_btn.state(["!disabled"])
                    self.scheme_var.set(SCHEME_BALANCED)
                    self._switch_scheme()
                elif item[0] == "error":
                    self.generate_btn.state(["!disabled"])
                    self.smart_btn.state(["!disabled"])
                    self.status_var.set("生成失败")
                    messagebox.showerror("生成失败", item[1])
        except queue.Empty:
            pass
        self._poll_after_id = self.after(80, self._poll_queue)

    def destroy(self) -> None:
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        super().destroy()

    def _fill_color_tree(self) -> None:
        self.color_tree.delete(*self.color_tree.get_children())
        if not self.result:
            return
        self.selected_color = min(max(0, self.selected_color), len(self.result.palette) - 1)
        counts = dict((c.code, n) for c, n in self.result.counts())
        for i, color in enumerate(self.result.palette):
            label = color.code if color.name == color.code else f"{color.code}  {color.name}"
            self.color_tree.insert("", "end", iid=str(i), values=(label, counts.get(color.code, 0)))
        if self.result.palette:
            self._set_selected_color(self.selected_color)

    def _rebuild_swatch_palette(self) -> None:
        for child in self.swatch_grid.winfo_children():
            child.destroy()
        self.swatch_buttons.clear()
        if not self.result:
            return

        columns = 4
        for i, color in enumerate(self.result.palette):
            tile = ttk.Frame(self.swatch_grid, padding=(3, 3))
            tile.grid(row=i // columns, column=i % columns, sticky="nsew", padx=2, pady=2)
            self.swatch_grid.grid_columnconfigure(i % columns, weight=1, uniform="swatches")
            hex_color = "#{:02x}{:02x}{:02x}".format(*color.rgb)
            button = tk.Button(tile, width=4, height=2, bg=hex_color,
                               activebackground=hex_color, relief="flat", bd=0,
                               highlightthickness=3, highlightbackground="#555b66",
                               highlightcolor=ACCENT, cursor="hand2",
                               command=lambda index=i: self._set_selected_color(index))
            button.pack(fill="x")
            label = ttk.Label(tile, text=color.code, anchor="center",
                              style="Muted.TLabel", font=("Microsoft YaHei UI", 8))
            label.pack(fill="x", pady=(2, 0))
            label.bind("<Button-1>", lambda _e, index=i: self._set_selected_color(index))
            button.bind("<MouseWheel>", self._scroll_swatches)
            label.bind("<MouseWheel>", self._scroll_swatches)
            self.swatch_buttons.append(button)

        self.swatch_canvas.yview_moveto(0)
        self._set_selected_color(self.selected_color)

    def _set_selected_color(self, index: int, sync_tree: bool = True) -> None:
        if not self.result or not self.result.palette:
            return
        index = min(max(0, int(index)), len(self.result.palette) - 1)
        self.selected_color = index
        color = self.result.palette[index]

        for i, button in enumerate(self.swatch_buttons):
            chosen = i == index
            button.configure(highlightbackground=ACCENT if chosen else "#555b66",
                             relief="sunken" if chosen else "flat")

        self.selected_chip.delete("all")
        hex_color = "#{:02x}{:02x}{:02x}".format(*color.rgb)
        self.selected_chip.create_rectangle(1, 1, 29, 29, fill=hex_color, outline=hex_color)
        label = color.code if color.name == color.code else f"{color.code}  {color.name}"
        self.selected_color_var.set(f"当前：{label}")

        if sync_tree and self.color_tree.exists(str(index)):
            self.color_tree.selection_set(str(index))
            self.color_tree.see(str(index))

    def _select_tree_color(self, _event=None) -> None:
        selected = self.color_tree.selection()
        if selected:
            self._set_selected_color(int(selected[0]), sync_tree=False)

    def _paint_event(self, event) -> None:
        if not self.result:
            return
        ox, oy, cell = self.display_info
        x, y = int((event.x - ox) // cell), int((event.y - oy) // cell)
        if not (0 <= x < self.result.width and 0 <= y < self.result.height):
            return
        old, new = int(self.result.indices[y, x]), self.selected_color
        if old == new:
            return
        self.result.indices[y, x] = new
        self.undo_stack.append((x, y, old, new))
        self.redo_stack.clear()
        self.refresh_pattern(); self._fill_color_tree(); self._refresh_info()

    def undo(self) -> None:
        if not self.result or not self.undo_stack:
            return
        x, y, old, new = self.undo_stack.pop()
        self.result.indices[y, x] = old
        self.redo_stack.append((x, y, old, new))
        self.refresh_pattern(); self._fill_color_tree(); self._refresh_info()

    def redo(self) -> None:
        if not self.result or not self.redo_stack:
            return
        x, y, old, new = self.redo_stack.pop()
        self.result.indices[y, x] = new
        self.undo_stack.append((x, y, old, new))
        self.refresh_pattern(); self._fill_color_tree(); self._refresh_info()

    def _refresh_info(self) -> None:
        if not self.result:
            return
        md = self.result.metadata
        mm = md["physical_mm_5mm"]
        boards = md["boards_29x29"]
        self.info_var.set(f"{md['total_beads']} 颗豆 · 约 {mm[0]/10:.1f} × {mm[1]/10:.1f} cm\n"
                          f"需 {boards[0]} × {boards[1]} 块 29×29 板 · 模式：{self.result.profile}")

    def export(self) -> None:
        if not self.result:
            messagebox.showinfo("尚未生成", "请先打开图片并生成图纸。")
            return
        initial = f"{(self.source_path.stem if self.source_path else 'pattern')}_MOSAIBEADS"
        folder = filedialog.askdirectory(title="选择导出文件夹", mustexist=True)
        if not folder:
            return
        out = Path(folder) / initial
        try:
            files = export_bundle(self.result, out, self.source_path.name if self.source_path else "pattern")
            messagebox.showinfo("导出完成", f"已导出 {len(files)} 个文件：\n{out}\n\n包含成品图、像素图、高清图纸、PDF、用量 CSV 和可编辑工程数据。")
            self.status_var.set(f"已导出到：{out}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))


def run() -> None:
    app = BeadSketchApp()
    app.mainloop()
