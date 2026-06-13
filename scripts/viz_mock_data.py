"""
Mock Data Visualizer for MicroDreamer
======================================
Interactive tkinter + matplotlib UI to inspect mock episodes.

Usage:
    python scripts/viz_mock_data.py [--data_dir DATA_DIR]

Features:
  - Episode browser (test_raw / viz_mock / any custom dir)
  - Frame-by-frame viewer with playback controls (◀ ▶ ⏮ ⏭)
  - Stage XY trajectory plot (2D path, colour-coded by subgoal)
  - Stage & Pipette positions vs time
  - 5-DOF action deltas vs time
  - Subgoal timeline bar
  - Frame statistics panel
  - Synchronised cursor across all plots
"""

import argparse
import json
import os
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle as MplCircle

# ─── colour palette for subgoal action_types ────────────────────────────────
ACTION_COLORS = {
    "observe": "#4fc3f7",
    "move_stage": "#81c784",
    "lower_pipette": "#ffb74d",
    "aspire": "#e57373",
    "inject": "#ba68c8",
    "release": "#fff176",
    "navigate": "#ce93d8",
}

DARK_BG = "#1e1e1e"
DARK_FG = "#d4d4d4"
DARK_PANEL = "#2b2b2b"
DARK_ACCENT = "#094771"


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def discover_episodes(data_dir: str) -> list[dict]:
    """Scan *data_dir* for episode folders (containing data.npz + metadata.json)."""
    episodes = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return episodes
    for entry in sorted(data_path.iterdir()):
        npz = entry / "data.npz"
        meta = entry / "metadata.json"
        if npz.exists() and meta.exists():
            episodes.append({"dir": str(entry), "name": entry.name})
    return episodes


def load_episode(ep_dir: str) -> dict:
    """Load an episode's npz data + metadata into a dict."""
    npz_path = os.path.join(ep_dir, "data.npz")
    meta_path = os.path.join(ep_dir, "metadata.json")

    data = np.load(npz_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Compute deltas (actions) from absolute positions
    stage_pos = data["stage_positions"]  # (T, 2)
    pip_pos = data["pipette_positions"]  # (T, 3)

    stage_deltas = np.diff(stage_pos, axis=0)  # (T-1, 2)
    pip_deltas = np.diff(pip_pos, axis=0)  # (T-1, 3)
    actions = np.concatenate([stage_deltas, pip_deltas], axis=1)  # (T-1, 5)

    return {
        "frames": data["frames"],  # (T, H, W) uint8
        "stage_positions": stage_pos,
        "pipette_positions": pip_pos,
        "timestamps": data["timestamps"],
        "actions": actions,  # (T-1, 5) computed deltas
        "metadata": meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dark-themed Matplotlib Figure helper
# ═══════════════════════════════════════════════════════════════════════════════

def make_dark_figure(dpi=100, nrows=1, ncols=1):
    fig = Figure(dpi=dpi, tight_layout=True, facecolor=DARK_BG)
    axes = fig.subplots(nrows, ncols)
    if not hasattr(axes, "__len__"):
        axes = np.array([axes])
    for ax in axes.flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#555")
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

class VizApp:
    """Mock data visualizer using tkinter + matplotlib."""

    def __init__(self, root: tk.Tk, data_dir: str):
        self.root = root
        self.root.title("MicroDreamer — Mock Data Viewer")
        self.root.geometry("1500x950")
        self.root.configure(bg=DARK_PANEL)

        self._data_dir = data_dir
        self._episodes: list[dict] = []
        self._episode_data: Optional[dict] = None
        self._playing = False
        self._play_after_id = None
        self._cursor_artists = []
        self._cursor_pos_lines = []
        self._cursor_act_line = None
        self._cursor_traj_dot = None

        self._apply_ttk_theme()
        self._build_ui()
        self._load_episode_list()

    # ────────────────────────────────────────────────────────────────────────
    # Theme
    # ────────────────────────────────────────────────────────────────────────

    def _apply_ttk_theme(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=DARK_PANEL, foreground=DARK_FG)
        style.configure("TLabel", background=DARK_PANEL, foreground=DARK_FG)
        style.configure("TButton", background="#3c3c3c", foreground=DARK_FG, padding=4)
        style.map("TButton", background=[("active", DARK_ACCENT)])
        style.configure("TCombobox", fieldbackground="#3c3c3c", foreground=DARK_FG)
        style.configure("TScale", background=DARK_PANEL)
        style.configure("TFrame", background=DARK_PANEL)
        style.configure("TLabelframe", background=DARK_PANEL, foreground=DARK_FG)
        style.configure("TLabelframe.Label", background=DARK_PANEL, foreground=DARK_FG)
        style.configure("Horizontal.TScale", background=DARK_PANEL)

    # ────────────────────────────────────────────────────────────────────────
    # UI layout
    # ────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar: dir + episode selectors ──
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))

        ttk.Label(top, text="Data Dir:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar()
        self.dir_combo = ttk.Combobox(top, textvariable=self.dir_var, width=40, state="readonly")
        self.dir_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.dir_combo.bind("<<ComboboxSelected>>", self._on_dir_changed)

        ttk.Label(top, text="Episode:").pack(side=tk.LEFT)
        self.ep_var = tk.StringVar()
        self.ep_combo = ttk.Combobox(top, textvariable=self.ep_var, width=50, state="readonly")
        self.ep_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.ep_combo.bind("<<ComboboxSelected>>", self._on_episode_changed)

        ttk.Button(top, text="↻ Reload", command=self._load_episode_list).pack(side=tk.LEFT)

        # ── Main content: left (frame+controls) | right (plots+meta) ──
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # --- Left panel: frame viewer + controls ---
        left = ttk.Frame(main_pw)
        main_pw.add(left, weight=3)

        # Frame canvas
        self.frame_canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.frame_canvas.pack(fill=tk.BOTH, expand=True)

        # Frame info
        self.frame_info = ttk.Label(left, text="No data loaded", anchor=tk.CENTER)
        self.frame_info.pack(fill=tk.X, pady=2)

        # Playback controls
        ctrl = ttk.Frame(left)
        ctrl.pack(fill=tk.X, pady=2)

        self.btn_start = ttk.Button(ctrl, text="⏮", width=3, command=lambda: self._goto(0))
        self.btn_prev = ttk.Button(ctrl, text="◀", width=3, command=lambda: self._goto(self._cur_frame - 1))
        self.btn_play = ttk.Button(ctrl, text="▶", width=3, command=self._toggle_play)
        self.btn_next = ttk.Button(ctrl, text="▶|", width=3, command=lambda: self._goto(self._cur_frame + 1))
        self.btn_end = ttk.Button(ctrl, text="⏭", width=3, command=lambda: self._goto(9999))

        self.btn_start.pack(side=tk.LEFT, padx=1)
        self.btn_prev.pack(side=tk.LEFT, padx=1)
        self.btn_play.pack(side=tk.LEFT, padx=1)
        self.btn_next.pack(side=tk.LEFT, padx=1)
        self.btn_end.pack(side=tk.LEFT, padx=1)

        self.slider_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(ctrl, from_=0, to=1, variable=self.slider_var,
                                orient=tk.HORIZONTAL, command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        ttk.Label(ctrl, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_spin = ttk.Spinbox(ctrl, from_=0.1, to=10.0, increment=0.1,
                                      textvariable=self.speed_var, width=5)
        self.speed_spin.pack(side=tk.LEFT, padx=2)

        # --- Right panel: tabs for plots + metadata ---
        right = ttk.Frame(main_pw)
        main_pw.add(right, weight=5)

        # Top-right: metadata + subgoal timeline (vertical paned)
        right_pw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_pw.pack(fill=tk.BOTH, expand=True)

        # --- Plot notebook ---
        self.plot_nb = ttk.Notebook(right_pw)
        right_pw.add(self.plot_nb, weight=3)

        # Tab: trajectory
        tab_traj = ttk.Frame(self.plot_nb)
        self.plot_nb.add(tab_traj, text=" Stage XY Path ")
        self.fig_traj, self.axes_traj = make_dark_figure(nrows=1, ncols=1)
        self.canvas_traj = FigureCanvasTkAgg(self.fig_traj, master=tab_traj)
        self.canvas_traj.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab: positions
        tab_pos = ttk.Frame(self.plot_nb)
        self.plot_nb.add(tab_pos, text=" Positions vs Time ")
        self.fig_pos, self.axes_pos = make_dark_figure(nrows=2, ncols=1)
        self.canvas_pos = FigureCanvasTkAgg(self.fig_pos, master=tab_pos)
        self.canvas_pos.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab: actions
        tab_act = ttk.Frame(self.plot_nb)
        self.plot_nb.add(tab_act, text=" Actions (Deltas) ")
        self.fig_act, self.axes_act = make_dark_figure(nrows=1, ncols=1)
        self.canvas_act = FigureCanvasTkAgg(self.fig_act, master=tab_act)
        self.canvas_act.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Bottom-right: metadata + subgoals + stats ---
        bottom = ttk.Frame(right_pw)
        right_pw.add(bottom, weight=2)

        bottom_pw = ttk.PanedWindow(bottom, orient=tk.HORIZONTAL)
        bottom_pw.pack(fill=tk.BOTH, expand=True)

        # Metadata text
        meta_frame = ttk.LabelFrame(bottom_pw, text="Episode Metadata")
        bottom_pw.add(meta_frame, weight=2)
        self.meta_text = tk.Text(meta_frame, bg=DARK_BG, fg=DARK_FG, font=("Consolas", 11),
                                  wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT)
        self.meta_text.pack(fill=tk.BOTH, expand=True)

        # Subgoal timeline
        sg_frame = ttk.LabelFrame(bottom_pw, text="Subgoal Timeline")
        bottom_pw.add(sg_frame, weight=1)
        self.fig_sg, self.axes_sg = make_dark_figure(dpi=80, nrows=1, ncols=1)
        self.canvas_sg = FigureCanvasTkAgg(self.fig_sg, master=sg_frame)
        self.canvas_sg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Stats text
        stats_frame = ttk.LabelFrame(bottom_pw, text="Frame Stats")
        bottom_pw.add(stats_frame, weight=1)
        self.stats_text = tk.Text(stats_frame, bg=DARK_BG, fg=DARK_FG, font=("Consolas", 10),
                                   wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Init frame state
        self._cur_frame = 0
        self._frames = None

    # ────────────────────────────────────────────────────────────────────────
    # Data directory & episode loading
    # ────────────────────────────────────────────────────────────────────────

    def _load_episode_list(self):
        project = Path(__file__).resolve().parent.parent  # MicroDreamer/
        known_dirs = [
            str(project / "data" / "test_raw"),
            str(project / "data" / "viz_mock"),
        ]
        if self._data_dir and self._data_dir not in known_dirs:
            known_dirs.insert(0, self._data_dir)

        existing = [d for d in known_dirs if Path(d).is_dir()]
        self.dir_combo["values"] = existing
        if existing:
            self.dir_combo.current(0)
            self._on_dir_changed(None)

    def _on_dir_changed(self, _event):
        d = self.dir_var.get()
        self._episodes = discover_episodes(d)
        names = [ep["name"] for ep in self._episodes]
        self.ep_combo["values"] = names
        if names:
            self.ep_combo.current(0)
            self._load_episode(0)

    def _on_episode_changed(self, _event):
        idx = self.ep_combo.current()
        if 0 <= idx < len(self._episodes):
            self._load_episode(idx)

    def _load_episode(self, idx: int):
        if idx < 0 or idx >= len(self._episodes):
            return
        ep_dir = self._episodes[idx]["dir"]
        self._episode_data = load_episode(ep_dir)
        data = self._episode_data
        self._frames = data["frames"]

        self._cur_frame = 0
        self.slider.configure(to=len(self._frames) - 1)
        self.slider_var.set(0)
        self._show_frame(0)
        self._show_metadata(data["metadata"])
        self._plot_all()
        self._update_stats(0)

    # ────────────────────────────────────────────────────────────────────────
    # Frame display
    # ────────────────────────────────────────────────────────────────────────

    def _show_frame(self, idx: int):
        if self._frames is None or len(self._frames) == 0:
            return
        idx = max(0, min(idx, len(self._frames) - 1))
        self._cur_frame = idx

        frame = self._frames[idx]
        h, w = frame.shape

        # Convert grayscale numpy → PhotoImage (via PPM for speed)
        # Pad to 3-channel
        rgb = np.stack([frame, frame, frame], axis=-1)  # (H, W, 3)
        ppm_header = f"P6\n{w} {h}\n255\n".encode()
        ppm_data = ppm_header + rgb.tobytes()

        # Use PIL for proper scaling if available, else raw
        cw = max(self.frame_canvas.winfo_width(), 200)
        ch = max(self.frame_canvas.winfo_height(), 160)
        scale = min(cw / w, ch / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        try:
            from PIL import Image, ImageTk
            img = Image.fromarray(rgb, "RGB")
            img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
        except ImportError:
            # Fallback: raw Tk photo (1:1, no scaling)
            self._photo = tk.PhotoImage(width=w, height=h)
            import base64
            b64 = base64.b64encode(ppm_data).decode()
            self._photo = tk.PhotoImage(data=b64)
            new_w, new_h = w, h

        self.frame_canvas.delete("all")
        cw = self.frame_canvas.winfo_width()
        ch = self.frame_canvas.winfo_height()
        self.frame_canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

        # Draw target crosshair on the frame
        target = self._episode_data["metadata"].get("target", {})
        if target and "pixel_position" in target:
            tx, ty = target["pixel_position"]
            tr = target.get("pixel_radius", 10)
            # Scale target position to canvas coordinates
            disp_w, disp_h = new_w, new_h
            ox = (cw - disp_w) // 2
            oy = (ch - disp_h) // 2
            sx = ox + int(tx * disp_w / w)
            sy = oy + int(ty * disp_h / h)
            sr = max(3, int(tr * disp_w / w))
            # Draw crosshair + circle
            self.frame_canvas.create_oval(sx - sr, sy - sr, sx + sr, sy + sr,
                                          outline="#ff5555", width=2, dash=(3, 3))
            self.frame_canvas.create_line(sx - sr - 4, sy, sx + sr + 4, sy,
                                          fill="#ff5555", width=1, dash=(2, 2))
            self.frame_canvas.create_line(sx, sy - sr - 4, sx, sy + sr + 4,
                                          fill="#ff5555", width=1, dash=(2, 2))
            self.frame_canvas.create_text(sx + sr + 6, sy - 4, text="TARGET",
                                          fill="#ff5555", font=("Arial", 8), anchor=tk.W)

        stage = self._episode_data["stage_positions"][idx]
        pip = self._episode_data["pipette_positions"][idx]
        ts = self._episode_data["timestamps"][idx]

        # Compute distances to target
        dist_info = ""
        if target:
            if "stage_position" in target:
                tsx, tsy = target["stage_position"]
                sd = ((stage[0] - tsx) ** 2 + (stage[1] - tsy) ** 2) ** 0.5
                dist_info += f"  |  Stage→Target: {sd:.1f}µm"
            if "pipette_position" in target:
                tpx, tpy, tpz = target["pipette_position"]
                pd = ((pip[0] - tpx) ** 2 + (pip[1] - tpy) ** 2 + (pip[2] - tpz) ** 2) ** 0.5
                dist_info += f"  |  Pip→Target: {pd:.1f}µm"

        self.frame_info.configure(
            text=f"Frame {idx}/{len(self._frames) - 1}  |  {w}×{h}  |  "
                 f"t={ts:.3f}s  |  Stage=({stage[0]:.1f}, {stage[1]:.1f})µm  |  "
                 f"Pip=({pip[0]:.1f}, {pip[1]:.1f}, {pip[2]:.1f})µm{dist_info}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Playback
    # ────────────────────────────────────────────────────────────────────────

    def _goto(self, idx):
        if self._frames is None:
            return
        idx = max(0, min(idx, len(self._frames) - 1))
        self.slider_var.set(idx)
        self._show_frame(idx)
        self._update_stats(idx)
        self._sync_cursors(idx)

    def _on_slider(self, val):
        idx = int(float(val))
        self._show_frame(idx)
        self._update_stats(idx)
        self._sync_cursors(idx)

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self._playing = True
        self.btn_play.configure(text="⏸")
        self._play_step()

    def _stop_play(self):
        self._playing = False
        self.btn_play.configure(text="▶")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_step(self):
        if not self._playing or self._frames is None:
            return
        nxt = self._cur_frame + 1
        if nxt >= len(self._frames):
            nxt = 0  # loop
        self._goto(nxt)
        interval = max(16, int(100 / self.speed_var.get()))
        self._play_after_id = self.root.after(interval, self._play_step)

    # ────────────────────────────────────────────────────────────────────────
    # Metadata display
    # ────────────────────────────────────────────────────────────────────────

    def _show_metadata(self, meta: dict):
        self.meta_text.configure(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)

        self.meta_text.insert(tk.END, f"Episode ID: {meta.get('episode_id', 'N/A')}\n")
        self.meta_text.insert(tk.END, f"Task: {meta.get('task_description', 'N/A')}\n")
        self.meta_text.insert(tk.END, f"Timestamp: {meta.get('timestamp', 'N/A')}\n")
        self.meta_text.insert(tk.END, f"Num Frames: {meta.get('num_frames', 'N/A')}\n")
        res = meta.get("resolution", "?")
        self.meta_text.insert(tk.END, f"Resolution: {res}\n\n")

        # Target info
        target = meta.get("target", {})
        if target:
            self.meta_text.insert(tk.END, "\n── Targets ──\n")
            if "stage_position" in target:
                tsx, tsy = target["stage_position"]
                self.meta_text.insert(tk.END, f"  Stage target: ({tsx:.1f}, {tsy:.1f}) µm\n")
            if "pipette_position" in target:
                tpx, tpy, tpz = target["pipette_position"]
                self.meta_text.insert(tk.END, f"  Pipette target: ({tpx:.1f}, {tpy:.1f}, {tpz:.1f}) µm\n")
            if "pixel_position" in target:
                px, py = target["pixel_position"]
                self.meta_text.insert(tk.END, f"  Pixel target: ({px}, {py}) r={target.get('pixel_radius', '?')}\n")

        start = meta.get("start_position", {})
        if start:
            self.meta_text.insert(tk.END, "\n── Start Positions ──\n")
            if "stage" in start:
                sx, sy = start["stage"]
                self.meta_text.insert(tk.END, f"  Stage start: ({sx:.1f}, {sy:.1f}) µm\n")
            if "pipette" in start:
                spx, spy, spz = start["pipette"]
                self.meta_text.insert(tk.END, f"  Pipette start: ({spx:.1f}, {spy:.1f}, {spz:.1f}) µm\n")

        subgoals = meta.get("subgoals", [])
        if subgoals:
            self.meta_text.insert(tk.END, f"\n── Subgoals ({len(subgoals)}) ──\n")
            for i, sg in enumerate(subgoals):
                desc = sg.get("description", "")
                atype = sg.get("action_type", "")
                sf = sg.get("start_frame", "?")
                ef = sg.get("end_frame", "?")
                self.meta_text.insert(tk.END, f"  [{i}] [{atype}] f{sf}-{ef}: {desc}\n")

        self.meta_text.configure(state=tk.DISABLED)

    # ────────────────────────────────────────────────────────────────────────
    # Statistics
    # ────────────────────────────────────────────────────────────────────────

    def _update_stats(self, frame_idx: int):
        data = self._episode_data
        if data is None:
            return
        frames = data["frames"]
        if frame_idx >= len(frames):
            return

        frame = frames[frame_idx]
        stage = data["stage_positions"][frame_idx]
        pip = data["pipette_positions"][frame_idx]
        ts = data["timestamps"][frame_idx]

        target = data["metadata"].get("target", {})

        lines = [
            f"── Frame {frame_idx} / {len(frames) - 1} ──",
            f"Time: {ts:.4f} s",
            f"Stage X: {stage[0]:+.2f} µm",
            f"Stage Y: {stage[1]:+.2f} µm",
            f"Pip X:   {pip[0]:+.2f} µm",
            f"Pip Y:   {pip[1]:+.2f} µm",
            f"Pip Z:   {pip[2]:+.2f} µm",
        ]

        # Distance to target
        if target:
            lines.append("")
            lines.append("── Distance to Target ──")
            if "stage_position" in target:
                tsx, tsy = target["stage_position"]
                sd = ((stage[0] - tsx) ** 2 + (stage[1] - tsy) ** 2) ** 0.5
                lines.append(f"  Stage: {sd:.2f} µm")
            if "pipette_position" in target:
                tpx, tpy, tpz = target["pipette_position"]
                pd_xy = ((pip[0] - tpx) ** 2 + (pip[1] - tpy) ** 2) ** 0.5
                pd_z = abs(pip[2] - tpz)
                pd = (pd_xy ** 2 + pd_z ** 2) ** 0.5
                lines.append(f"  Pip XY: {pd_xy:.2f} µm")
                lines.append(f"  Pip Z:  {pd_z:.2f} µm")
                lines.append(f"  Pip 3D: {pd:.2f} µm")
                # Check if at target
                if sd < 5 and pd < 10:
                    lines.append("  ✅ AT TARGET")

        lines += [
            "",
            f"── Pixel Stats ──",
            f"Min:   {frame.min()}",
            f"Max:   {frame.max()}",
            f"Mean:  {frame.mean():.1f}",
            f"Std:   {frame.std():.1f}",
        ]

        if frame_idx > 0 and frame_idx - 1 < len(data["actions"]):
            act = data["actions"][frame_idx - 1]
            lines += [
                "",
                "── Action Δ ──",
                f"dStgX: {act[0]:+.4f} µm",
                f"dStgY: {act[1]:+.4f} µm",
                f"dPipX: {act[2]:+.4f} µm",
                f"dPipY: {act[3]:+.4f} µm",
                f"dPipZ: {act[4]:+.4f} µm",
            ]

        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "\n".join(lines))
        self.stats_text.configure(state=tk.DISABLED)

    # ────────────────────────────────────────────────────────────────────────
    # Plotting
    # ────────────────────────────────────────────────────────────────────────

    def _plot_all(self):
        data = self._episode_data
        if data is None:
            return
        self._cursor_pos_lines = []
        self._cursor_act_line = None
        self._cursor_traj_dot = None
        self._plot_trajectory(data)
        self._plot_positions(data)
        self._plot_actions(data)
        self._plot_subgoals(data)

    def _plot_trajectory(self, data: dict):
        ax = self.axes_traj[0]
        ax.clear()
        ax.set_facecolor(DARK_BG)

        stage = data["stage_positions"]
        subgoals = data["metadata"].get("subgoals", [])
        T = len(stage)
        seg_len = T // max(len(subgoals), 1)

        # Draw colour-coded segments
        for i, sg in enumerate(subgoals):
            s = i * seg_len
            e = (i + 1) * seg_len if i < len(subgoals) - 1 else T
            atype = sg.get("action_type", "")
            c = ACTION_COLORS.get(atype, "#888")
            ax.plot(stage[s:e + 1, 0], stage[s:e + 1, 1], color=c, linewidth=2.5, alpha=0.7)

        # Full path (thin)
        ax.plot(stage[:, 0], stage[:, 1], color="#4fc3f7", linewidth=0.8, alpha=0.5)

        # Start / End markers
        ax.scatter(stage[0, 0], stage[0, 1], c="#81c784", s=80, zorder=5, marker="o", label="Start")
        ax.scatter(stage[-1, 0], stage[-1, 1], c="#e57373", s=80, zorder=5, marker="s", label="End")

        # Target marker
        target = data["metadata"].get("target", {})
        if "stage_position" in target:
            tx, ty = target["stage_position"]
            ax.scatter(tx, ty, c="#ff5555", s=200, zorder=6, marker="*", linewidths=1.5,
                       edgecolors="white", label="TARGET")
            # Target tolerance circle
            circle = MplCircle((tx, ty), 5, fill=False, color="#ff5555",
                               linestyle="--", linewidth=1, alpha=0.5)
            ax.add_patch(circle)

        # Start position marker
        start = data["metadata"].get("start_position", {})
        if "stage" in start:
            sx, sy = start["stage"]
            ax.scatter(sx, sy, c="#81c784", s=80, zorder=5, marker="D",
                       edgecolors="white", linewidths=1, label="Start Pos")

        # Cursor dot (will be updated by _sync_cursors)
        self._cursor_traj_dot = ax.scatter([], [], c="#ffffff", s=120, zorder=10, marker="x", linewidths=2)

        ax.set_xlabel("Stage X (µm)", color="#aaa", fontsize=8)
        ax.set_ylabel("Stage Y (µm)", color="#aaa", fontsize=8)
        ax.set_title("Stage XY Trajectory (colour = subgoal)", color=DARK_FG, fontsize=9)
        ax.legend(fontsize=7, facecolor=DARK_BG, edgecolor="#555", labelcolor=DARK_FG)
        ax.set_aspect("equal", adjustable="datalim")
        self.canvas_traj.draw()

    def _plot_positions(self, data: dict):
        axes = self.axes_pos
        for ax in axes:
            ax.clear()
            ax.set_facecolor(DARK_BG)

        ts = data["timestamps"]
        stage = data["stage_positions"]
        pip = data["pipette_positions"]

        # Target info
        target = data["metadata"].get("target", {})

        # Stage XY
        axes[0].plot(ts, stage[:, 0], label="Stage X", color="#4fc3f7", linewidth=1)
        axes[0].plot(ts, stage[:, 1], label="Stage Y", color="#81c784", linewidth=1)
        # Target stage lines
        if "stage_position" in target:
            tsx, tsy = target["stage_position"]
            axes[0].axhline(tsx, color="#4fc3f7", linestyle="--", linewidth=1, alpha=0.4, label=f"Target X={tsx:.0f}")
            axes[0].axhline(tsy, color="#81c784", linestyle="--", linewidth=1, alpha=0.4, label=f"Target Y={tsy:.0f}")
        axes[0].set_ylabel("Stage (µm)", color="#aaa", fontsize=8)
        axes[0].set_title("Stage & Pipette Positions vs Time", color=DARK_FG, fontsize=9)
        axes[0].legend(fontsize=6, facecolor=DARK_BG, edgecolor="#555", labelcolor=DARK_FG, ncol=3)

        # Pipette XYZ
        axes[1].plot(ts, pip[:, 0], label="Pip X", color="#ffb74d", linewidth=1)
        axes[1].plot(ts, pip[:, 1], label="Pip Y", color="#e57373", linewidth=1)
        axes[1].plot(ts, pip[:, 2], label="Pip Z", color="#ba68c8", linewidth=1)
        # Target pipette lines
        if "pipette_position" in target:
            tpx, tpy, tpz = target["pipette_position"]
            axes[1].axhline(tpx, color="#ffb74d", linestyle="--", linewidth=1, alpha=0.4, label=f"Tgt X={tpx:.0f}")
            axes[1].axhline(tpy, color="#e57373", linestyle="--", linewidth=1, alpha=0.4, label=f"Tgt Y={tpy:.0f}")
            axes[1].axhline(tpz, color="#ba68c8", linestyle="--", linewidth=1, alpha=0.4, label=f"Tgt Z={tpz:.0f}")
        axes[1].set_ylabel("Pipette (µm)", color="#aaa", fontsize=8)
        axes[1].set_xlabel("Time (s)", color="#aaa", fontsize=8)
        axes[1].legend(fontsize=6, facecolor=DARK_BG, edgecolor="#555", labelcolor=DARK_FG, ncol=3)

        # Subgoal shading
        subgoals = data["metadata"].get("subgoals", [])
        T = len(ts)
        seg_len = T // max(len(subgoals), 1)
        for ax in axes:
            for i, sg in enumerate(subgoals):
                s = i * seg_len
                e = (i + 1) * seg_len if i < len(subgoals) - 1 else T
                atype = sg.get("action_type", "")
                c = ACTION_COLORS.get(atype, "#888")
                ax.axvspan(ts[s], ts[min(e, T - 1)], alpha=0.12, color=c)
            # Cursor lines
            self._cursor_pos_lines.append(ax.axvline(0, color="#fff", linewidth=1.5, linestyle="--", alpha=0.6, visible=False))

        self.canvas_pos.draw()

    def _plot_actions(self, data: dict):
        ax = self.axes_act[0]
        ax.clear()
        ax.set_facecolor(DARK_BG)

        actions = data["actions"]  # (T-1, 5)
        ts = data["timestamps"][1:]
        labels = ["Stage dX", "Stage dY", "Pip dX", "Pip dY", "Pip dZ"]
        colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8"]

        for i in range(5):
            ax.plot(ts, actions[:, i], label=labels[i], color=colors[i], linewidth=1, alpha=0.85)

        # Cursor
        self._cursor_act_line = ax.axvline(0, color="#fff", linewidth=1.5, linestyle="--", alpha=0.6, visible=False)

        ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax.set_ylabel("Delta (µm)", color="#aaa", fontsize=8)
        ax.set_title("5-DOF Action Deltas", color=DARK_FG, fontsize=9)
        ax.legend(fontsize=7, facecolor=DARK_BG, edgecolor="#555", labelcolor=DARK_FG, ncol=3)
        self.canvas_act.draw()

    def _plot_subgoals(self, data: dict):
        ax = self.axes_sg[0]
        ax.clear()
        ax.set_facecolor(DARK_BG)

        subgoals = data["metadata"].get("subgoals", [])
        T = len(data["frames"])
        seg_len = T // max(len(subgoals), 1)

        for i, sg in enumerate(subgoals):
            s = i * seg_len
            e = (i + 1) * seg_len if i < len(subgoals) - 1 else T
            atype = sg.get("action_type", "unknown")
            desc = sg.get("description", "")
            c = ACTION_COLORS.get(atype, "#888")
            ax.barh(0, e - s, left=s, height=0.6, color=c, edgecolor="white", linewidth=0.5)
            mid = (s + e) / 2
            label = f"{atype}\n{desc[:25]}"
            ax.text(mid, 0, label, ha="center", va="center", fontsize=6, color="white", weight="bold")

        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Frame Index", color="#aaa", fontsize=7)
        ax.set_yticks([])
        ax.set_title("Subgoal Timeline", color=DARK_FG, fontsize=8)
        self.canvas_sg.draw()

    # ────────────────────────────────────────────────────────────────────────
    # Cursor synchronisation
    # ────────────────────────────────────────────────────────────────────────

    def _sync_cursors(self, frame_idx: int):
        data = self._episode_data
        if data is None:
            return
        ts = data["timestamps"]
        stage = data["stage_positions"]
        t = ts[frame_idx] if frame_idx < len(ts) else ts[-1]

        # Trajectory dot
        if hasattr(self, "_cursor_traj_dot") and frame_idx < len(stage):
            self._cursor_traj_dot.set_offsets([[stage[frame_idx, 0], stage[frame_idx, 1]]])

        # Position cursor lines
        if hasattr(self, "_cursor_pos_lines"):
            for line in self._cursor_pos_lines:
                line.set_xdata([t, t])
                line.set_visible(True)

        # Action cursor
        if hasattr(self, "_cursor_act_line") and frame_idx > 0:
            self._cursor_act_line.set_xdata([t, t])
            self._cursor_act_line.set_visible(True)

        # Redraw (draw_idle is faster than draw)
        try:
            self.canvas_traj.draw_idle()
            self.canvas_pos.draw_idle()
            self.canvas_act.draw_idle()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MicroDreamer Mock Data Visualizer")
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to episode directory (default: auto-discover test_raw & viz_mock)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        project = Path(__file__).resolve().parent.parent
        data_dir = str(project / "data" / "test_raw")

    root = tk.Tk()
    app = VizApp(root, data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
