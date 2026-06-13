"""MicroDreamer Data Collection UI.

Tkinter-based GUI for microscope data collection with:
- Live camera preview
- Stage control (manual + PID auto-positioning)
- Pipette control (manual + Z-axis)
- Data recording with task descriptions
"""

import sys
import os
import time
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from hardware.factory import create_camera, create_stage, create_pipette
from hardware.base import CameraBase, StageBase, PipetteBase


class PIDController:
    """PID controller for auto-positioning."""

    def __init__(self, kp=0.5, ki=0.01, kd=0.1, output_limit=50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = None

    def update(self, error: float) -> float:
        now = time.time()
        dt = (now - self._last_time) if self._last_time else 0.01
        self._last_time = now

        self._integral += error * dt
        # Anti-windup
        self._integral = max(-self.output_limit, min(self.output_limit, self._integral))

        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class DataCollector:
    """Handles data collection and saving."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.frames = []
        self.stage_positions = []
        self.pipette_positions = []
        self.timestamps = []
        self.task_description = ""
        self.start_time = None

    def start_recording(self, task_description: str = ""):
        self.recording = True
        self.frames = []
        self.stage_positions = []
        self.pipette_positions = []
        self.timestamps = []
        self.task_description = task_description
        self.start_time = time.time()

    def add_sample(self, frame: np.ndarray, stage_pos: tuple, pipette_pos: tuple):
        if not self.recording:
            return
        self.frames.append(frame.copy())
        self.stage_positions.append(stage_pos)
        self.pipette_positions.append(pipette_pos)
        self.timestamps.append(time.time())

    def stop_recording(self) -> Optional[str]:
        if not self.recording:
            return None
        self.recording = False

        if len(self.frames) == 0:
            return None

        # Create episode directory
        episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        episode_dir = self.output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        frames_arr = np.array(self.frames, dtype=np.uint8)
        stage_arr = np.array(self.stage_positions, dtype=np.float32)
        pipette_arr = np.array(self.pipette_positions, dtype=np.float32)

        np.savez_compressed(
            episode_dir / "data.npz",
            frames=frames_arr,
            stage_positions=stage_arr,
            pipette_positions=pipette_arr,
        )

        # Save metadata
        metadata = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "task_description": self.task_description,
            "num_frames": len(self.frames),
            "duration_seconds": time.time() - self.start_time,
            "camera_fps": len(self.frames) / max(time.time() - self.start_time, 0.001),
            "camera_resolution": list(frames_arr.shape[1:][::-1]) if len(frames_arr.shape) >= 3 else [1600, 1200],
        }
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return str(episode_dir)

    @property
    def frame_count(self):
        return len(self.frames)


class CollectionUI:
    """Main data collection UI."""

    def __init__(self, config_path: str = None):
        self.cfg = load_config(config_path)

        # Initialize hardware
        self.camera = create_camera(self.cfg)
        self.stage = create_stage(self.cfg)
        self.pipette = create_pipette(self.cfg)

        # PID controllers for X and Y
        self.pid_x = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limit=100.0)
        self.pid_y = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limit=100.0)
        self.pid_enabled = False
        self.pid_target = None  # (pixel_x, pixel_y) target in camera view

        # Data collector
        self.collector = DataCollector()

        # State
        self.connected = False
        self.preview_running = False
        self.current_frame = None
        self.fps_counter = deque(maxlen=30)
        self.last_capture_time = 0

        # Build UI
        self.root = tk.Tk()
        self.root.title("MicroDreamer Data Collection")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

    def _build_ui(self):
        """Build the UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Connection controls
        self._build_connection_bar(main_frame)

        # Middle: Left=Camera, Right=Controls
        middle = ttk.Frame(main_frame)
        middle.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left: Camera canvas
        self._build_camera_panel(middle)

        # Right: Control panels
        right = ttk.Frame(middle, width=350)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right.pack_propagate(False)

        self._build_stage_panel(right)
        self._build_pipette_panel(right)
        self._build_pid_panel(right)
        self._build_recording_panel(right)

        # Bottom: Status bar
        self._build_status_bar(main_frame)

    def _build_connection_bar(self, parent):
        """Connection controls at top."""
        frame = ttk.LabelFrame(parent, text="连接 / Connection", padding=5)
        frame.pack(fill=tk.X, pady=2)

        ttk.Button(frame, text="连接设备", command=self._connect_devices).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="断开设备", command=self._disconnect_devices).pack(side=tk.LEFT, padx=5)

        self.conn_status = ttk.Label(frame, text="● 未连接", foreground="red")
        self.conn_status.pack(side=tk.LEFT, padx=20)

        ttk.Button(frame, text="刷新预览", command=self._toggle_preview).pack(side=tk.RIGHT, padx=5)

    def _build_camera_panel(self, parent):
        """Camera preview on the left."""
        frame = ttk.LabelFrame(parent, text="相机预览 / Camera Preview", padding=5)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas for camera view
        self.canvas = tk.Canvas(frame, bg="black", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Canvas info
        self.canvas_info = ttk.Label(frame, text="点击画面设置 PID 目标位置 / Click to set PID target")
        self.canvas_info.pack()

    def _build_stage_panel(self, parent):
        """Stage control panel."""
        frame = ttk.LabelFrame(parent, text="Stage 控制", padding=5)
        frame.pack(fill=tk.X, pady=2)

        # Position display
        pos_frame = ttk.Frame(frame)
        pos_frame.pack(fill=tk.X)
        ttk.Label(pos_frame, text="X:").pack(side=tk.LEFT)
        self.stage_x_var = tk.StringVar(value="0.0")
        ttk.Label(pos_frame, textvariable=self.stage_x_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(pos_frame, text="μm").pack(side=tk.LEFT)
        ttk.Label(pos_frame, text="Y:").pack(side=tk.LEFT, padx=(10, 0))
        self.stage_y_var = tk.StringVar(value="0.0")
        ttk.Label(pos_frame, textvariable=self.stage_y_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(pos_frame, text="μm").pack(side=tk.LEFT)

        # Move controls
        move_frame = ttk.Frame(frame)
        move_frame.pack(fill=tk.X, pady=5)

        ttk.Label(move_frame, text="步长:").pack(side=tk.LEFT)
        self.stage_step_var = tk.StringVar(value="10")
        step_entry = ttk.Entry(move_frame, textvariable=self.stage_step_var, width=8)
        step_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(move_frame, text="μm").pack(side=tk.LEFT)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="↑", width=4,
                   command=lambda: self._move_stage(0, 1)).grid(row=0, column=1)
        ttk.Button(btn_frame, text="←", width=4,
                   command=lambda: self._move_stage(-1, 0)).grid(row=1, column=0)
        ttk.Button(btn_frame, text="■", width=4,
                   command=self._stop_stage).grid(row=1, column=1)
        ttk.Button(btn_frame, text="→", width=4,
                   command=lambda: self._move_stage(1, 0)).grid(row=1, column=2)
        ttk.Button(btn_frame, text="↓", width=4,
                   command=lambda: self._move_stage(0, -1)).grid(row=2, column=1)

        # Absolute move
        abs_frame = ttk.Frame(frame)
        abs_frame.pack(fill=tk.X, pady=2)
        ttk.Label(abs_frame, text="绝对位置:").pack(side=tk.LEFT)
        self.abs_x_var = tk.StringVar(value="0")
        self.abs_y_var = tk.StringVar(value="0")
        ttk.Entry(abs_frame, textvariable=self.abs_x_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Entry(abs_frame, textvariable=self.abs_y_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(abs_frame, text="移动", command=self._move_stage_absolute).pack(side=tk.LEFT, padx=2)

    def _build_pipette_panel(self, parent):
        """Pipette control panel."""
        frame = ttk.LabelFrame(parent, text="Pipette 控制", padding=5)
        frame.pack(fill=tk.X, pady=2)

        # Position display
        pos_frame = ttk.Frame(frame)
        pos_frame.pack(fill=tk.X)
        for axis, var_name in [("X", "pip_x"), ("Y", "pip_y"), ("Z", "pip_z")]:
            ttk.Label(pos_frame, text=f"{axis}:").pack(side=tk.LEFT, padx=(5, 0))
            var = tk.StringVar(value="0.0")
            setattr(self, f"{var_name}_var", var)
            ttk.Label(pos_frame, textvariable=var, width=6).pack(side=tk.LEFT, padx=2)

        # Z-axis controls (most important for pipette)
        z_frame = ttk.LabelFrame(frame, text="Z 轴控制", padding=3)
        z_frame.pack(fill=tk.X, pady=3)

        z_btn_frame = ttk.Frame(z_frame)
        z_btn_frame.pack(fill=tk.X)

        ttk.Label(z_btn_frame, text="步长:").pack(side=tk.LEFT)
        self.pip_z_step_var = tk.StringVar(value="5")
        ttk.Entry(z_btn_frame, textvariable=self.pip_z_step_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(z_btn_frame, text="μm").pack(side=tk.LEFT)

        ttk.Button(z_btn_frame, text="▲ 上升", width=8,
                   command=lambda: self._move_pipette_z(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(z_btn_frame, text="▼ 下降", width=8,
                   command=lambda: self._move_pipette_z(-1)).pack(side=tk.LEFT, padx=5)

        # XY move
        xy_frame = ttk.Frame(frame)
        xy_frame.pack(fill=tk.X, pady=2)
        ttk.Label(xy_frame, text="XY 步长:").pack(side=tk.LEFT)
        self.pip_xy_step_var = tk.StringVar(value="5")
        ttk.Entry(xy_frame, textvariable=self.pip_xy_step_var, width=5).pack(side=tk.LEFT, padx=2)

        xy_btn_frame = ttk.Frame(frame)
        xy_btn_frame.pack(fill=tk.X)
        ttk.Button(xy_btn_frame, text="↑", width=4,
                   command=lambda: self._move_pipette_xy(0, 1)).grid(row=0, column=1)
        ttk.Button(xy_btn_frame, text="←", width=4,
                   command=lambda: self._move_pipette_xy(-1, 0)).grid(row=1, column=0)
        ttk.Button(xy_btn_frame, text="→", width=4,
                   command=lambda: self._move_pipette_xy(1, 0)).grid(row=1, column=2)
        ttk.Button(xy_btn_frame, text="↓", width=4,
                   command=lambda: self._move_pipette_xy(0, -1)).grid(row=2, column=1)

    def _build_pid_panel(self, parent):
        """PID auto-positioning panel."""
        frame = ttk.LabelFrame(parent, text="PID 自动定位", padding=5)
        frame.pack(fill=tk.X, pady=2)

        self.pid_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="启用 PID", variable=self.pid_enabled_var,
                        command=self._toggle_pid).pack(fill=tk.X)

        # PID parameters
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill=tk.X, pady=2)

        ttk.Label(param_frame, text="Kp:").grid(row=0, column=0)
        self.pid_kp_var = tk.StringVar(value="0.5")
        ttk.Entry(param_frame, textvariable=self.pid_kp_var, width=6).grid(row=0, column=1, padx=2)

        ttk.Label(param_frame, text="Ki:").grid(row=0, column=2)
        self.pid_ki_var = tk.StringVar(value="0.01")
        ttk.Entry(param_frame, textvariable=self.pid_ki_var, width=6).grid(row=0, column=3, padx=2)

        ttk.Label(param_frame, text="Kd:").grid(row=1, column=0)
        self.pid_kd_var = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.pid_kd_var, width=6).grid(row=1, column=1, padx=2)

        ttk.Label(param_frame, text="限幅:").grid(row=1, column=2)
        self.pid_limit_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.pid_limit_var, width=6).grid(row=1, column=3, padx=2)

        ttk.Button(param_frame, text="应用参数", command=self._apply_pid_params).grid(row=2, column=0, columnspan=4, pady=3)

        # Target display
        self.pid_target_label = ttk.Label(frame, text="目标: 未设置")
        self.pid_target_label.pack()

        ttk.Button(frame, text="清除目标", command=self._clear_pid_target).pack(fill=tk.X, pady=2)

    def _build_recording_panel(self, parent):
        """Data recording controls."""
        frame = ttk.LabelFrame(parent, text="数据采集", padding=5)
        frame.pack(fill=tk.X, pady=2)

        # Task description
        ttk.Label(frame, text="任务描述:").pack(anchor=tk.W)
        self.task_var = tk.StringVar(value="move to cell")
        task_entry = ttk.Entry(frame, textvariable=self.task_var, width=40)
        task_entry.pack(fill=tk.X, pady=2)

        # Recording controls
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.record_btn = ttk.Button(btn_frame, text="● 开始录制", command=self._toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.record_status = ttk.Label(btn_frame, text="未录制", foreground="gray")
        self.record_status.pack(side=tk.LEFT, padx=5)

        # Frame count
        self.frame_count_var = tk.StringVar(value="帧数: 0")
        ttk.Label(frame, textvariable=self.frame_count_var).pack(anchor=tk.W)

        # Save directory
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dir_frame, text="保存目录:").pack(side=tk.LEFT)
        self.save_dir_var = tk.StringVar(value="data/raw")
        ttk.Entry(dir_frame, textvariable=self.save_dir_var, width=25).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_frame, text="浏览", command=self._browse_save_dir).pack(side=tk.LEFT)

        # Episodes list
        ttk.Label(frame, text="已采集 Episodes:").pack(anchor=tk.W, pady=(5, 0))
        self.episode_list = tk.Listbox(frame, height=4, width=40)
        self.episode_list.pack(fill=tk.X)
        self._refresh_episode_list()

    def _build_status_bar(self, parent):
        """Status bar at bottom."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        self.fps_label = ttk.Label(frame, text="FPS: --")
        self.fps_label.pack(side=tk.LEFT, padx=10)

        self.pos_label = ttk.Label(frame, text="Stage: (0, 0) | Pipette: (0, 0, 0)")
        self.pos_label.pack(side=tk.LEFT, padx=10)

        self.msg_label = ttk.Label(frame, text="就绪")
        self.msg_label.pack(side=tk.RIGHT, padx=10)

    # ---- Hardware Actions ----

    def _connect_devices(self):
        """Connect all hardware devices."""
        try:
            if not self.camera.is_open():
                self.camera.open()
            if not self.stage.is_connected():
                self.stage.connect()
            if not self.pipette.is_connected():
                self.pipette.connect()
            self.connected = True
            self.conn_status.config(text="● 已连接", foreground="green")
            self._set_msg("设备已连接")
            self._start_preview()
        except Exception as e:
            messagebox.showerror("连接错误", str(e))

    def _disconnect_devices(self):
        """Disconnect all hardware devices."""
        self._stop_preview()
        try:
            if self.camera.is_open():
                self.camera.close()
            if self.stage.is_connected():
                self.stage.disconnect()
            if self.pipette.is_connected():
                self.pipette.disconnect()
        except Exception:
            pass
        self.connected = False
        self.conn_status.config(text="● 未连接", foreground="red")
        self._set_msg("设备已断开")

    def _move_stage(self, dx_sign, dy_sign):
        """Move stage by step in given direction."""
        if not self.stage.is_connected():
            return
        try:
            step = float(self.stage_step_var.get())
            self.stage.move_relative(dx_sign * step, dy_sign * step)
        except ValueError:
            pass

    def _move_stage_absolute(self):
        """Move stage to absolute position."""
        if not self.stage.is_connected():
            return
        try:
            x = float(self.abs_x_var.get())
            y = float(self.abs_y_var.get())
            self.stage.move_absolute(x, y)
        except ValueError:
            pass

    def _stop_stage(self):
        """Stop stage movement (no-op for virtual, real implementation varies)."""
        self._set_msg("Stage 已停止")

    def _move_pipette_z(self, sign):
        """Move pipette Z axis."""
        if not self.pipette.is_connected():
            return
        try:
            step = float(self.pip_z_step_var.get())
            self.pipette.move_relative(0, 0, sign * step)
        except ValueError:
            pass

    def _move_pipette_xy(self, dx_sign, dy_sign):
        """Move pipette XY."""
        if not self.pipette.is_connected():
            return
        try:
            step = float(self.pip_xy_step_var.get())
            self.pipette.move_relative(dx_sign * step, dy_sign * step, 0)
        except ValueError:
            pass

    # ---- PID ----

    def _toggle_pid(self):
        self.pid_enabled = self.pid_enabled_var.get()
        if self.pid_enabled:
            self._apply_pid_params()
            self._set_msg("PID 已启用")
        else:
            self.pid_x.reset()
            self.pid_y.reset()
            self._set_msg("PID 已禁用")

    def _apply_pid_params(self):
        try:
            self.pid_x.kp = float(self.pid_kp_var.get())
            self.pid_x.ki = float(self.pid_ki_var.get())
            self.pid_x.kd = float(self.pid_kd_var.get())
            self.pid_x.output_limit = float(self.pid_limit_var.get())
            self.pid_y.kp = float(self.pid_kp_var.get())
            self.pid_y.ki = float(self.pid_ki_var.get())
            self.pid_y.kd = float(self.pid_kd_var.get())
            self.pid_y.output_limit = float(self.pid_limit_var.get())
            self._set_msg("PID 参数已更新")
        except ValueError:
            messagebox.showerror("参数错误", "PID 参数格式不正确")

    def _on_canvas_click(self, event):
        """Handle canvas click - set PID target."""
        if not self.pid_enabled:
            return
        # Map canvas coordinates to image coordinates
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        # Image is displayed scaled to canvas
        self.pid_target = (event.x, event.y)
        self.pid_target_label.config(text=f"目标: ({event.x}, {event.y}) px")
        self._set_msg(f"PID 目标已设置: ({event.x}, {event.y})")

    def _clear_pid_target(self):
        self.pid_target = None
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_target_label.config(text="目标: 未设置")

    def _pid_step(self):
        """Run one PID step if enabled and target is set."""
        if not self.pid_enabled or self.pid_target is None:
            return
        if not self.stage.is_connected() or self.current_frame is None:
            return

        # Current position in image (center of frame as proxy)
        # In a real system, this would use object detection to find the target object
        # For now, we use the frame center as the "current position"
        h, w = self.current_frame.shape[:2]
        current_x = w / 2
        current_y = h / 2

        # Error: target - current (in pixels)
        error_x = self.pid_target[0] - current_x
        error_y = self.pid_target[1] - current_y

        # Convert pixel error to stage movement (um)
        pixel_size = self.cfg.get("pixel_size_um", 0.6)
        error_x_um = error_x * pixel_size
        error_y_um = -error_y * pixel_size  # Y axis inverted

        # PID output
        dx = self.pid_x.update(error_x_um)
        dy = self.pid_y.update(error_y_um)

        # Move stage
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            self.stage.move_relative(dx, dy)

    # ---- Recording ----

    def _toggle_recording(self):
        if self.collector.recording:
            # Stop recording
            episode_path = self.collector.stop_recording()
            self.record_btn.config(text="● 开始录制")
            self.record_status.config(text="未录制", foreground="gray")
            if episode_path:
                self._set_msg(f"已保存: {episode_path}")
                self._refresh_episode_list()
        else:
            # Start recording
            task = self.task_var.get().strip()
            if not task:
                messagebox.showwarning("警告", "请输入任务描述")
                return
            self.collector.start_recording(task)
            self.record_btn.config(text="■ 停止录制")
            self.record_status.config(text="● 录制中", foreground="red")
            self._set_msg("开始录制...")

    def _browse_save_dir(self):
        d = filedialog.askdirectory(initialdir=self.save_dir_var.get())
        if d:
            self.save_dir_var.set(d)
            self.collector.output_dir = Path(d)
            self._refresh_episode_list()

    def _refresh_episode_list(self):
        self.episode_list.delete(0, tk.END)
        save_dir = Path(self.save_dir_var.get())
        if save_dir.exists():
            episodes = sorted(save_dir.iterdir())
            for ep in episodes:
                if ep.is_dir() and (ep / "metadata.json").exists():
                    with open(ep / "metadata.json") as f:
                        meta = json.load(f)
                    self.episode_list.insert(tk.END,
                        f"{ep.name} | {meta.get('num_frames', '?')}帧 | {meta.get('task_description', '')}")

    # ---- Preview ----

    def _start_preview(self):
        if self.preview_running:
            return
        self.preview_running = True
        self._update_preview()

    def _stop_preview(self):
        self.preview_running = False

    def _toggle_preview(self):
        if self.preview_running:
            self._stop_preview()
        else:
            self._start_preview()

    def _update_preview(self):
        """Update camera preview (called periodically)."""
        if not self.preview_running:
            return

        try:
            if self.camera.is_open():
                frame_data = self.camera.capture()
                self.current_frame = frame_data.image

                # Run PID step
                self._pid_step()

                # Record sample
                if self.collector.recording and self.current_frame is not None:
                    stage_pos = self.stage.get_position() if self.stage.is_connected() else None
                    pipette_pos = self.pipette.get_position() if self.pipette.is_connected() else None
                    if stage_pos and pipette_pos:
                        self.collector.add_sample(
                            self.current_frame,
                            (stage_pos.x, stage_pos.y),
                            (pipette_pos.x, pipette_pos.y, pipette_pos.z),
                        )
                        self.frame_count_var.set(f"帧数: {self.collector.frame_count}")

                # Display frame
                self._display_frame(self.current_frame)

                # Update position displays
                self._update_position_display()

                # FPS
                now = time.time()
                self.fps_counter.append(now)
                if len(self.fps_counter) > 1:
                    fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
        except Exception as e:
            self._set_msg(f"预览错误: {e}")

        # Schedule next update (~30fps)
        self.root.after(33, self._update_preview)

    def _display_frame(self, frame: np.ndarray):
        """Display frame on canvas."""
        from PIL import Image, ImageTk

        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 800, 600

        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Convert to PIL Image
        if len(frame.shape) == 2:
            img = Image.fromarray(frame, mode='L')
        else:
            img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Convert to PhotoImage
        self._photo = ImageTk.PhotoImage(img)

        # Draw on canvas
        self.canvas.delete("all")
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self._photo)

        # Draw PID target crosshair
        if self.pid_target is not None:
            tx, ty = self.pid_target
            self.canvas.create_line(tx - 15, ty, tx + 15, ty, fill="red", width=2)
            self.canvas.create_line(tx, ty - 15, tx, ty + 15, fill="red", width=2)
            self.canvas.create_oval(tx - 20, ty - 20, tx + 20, ty + 20, outline="red", width=1)

    def _update_position_display(self):
        """Update position labels."""
        if self.stage.is_connected():
            pos = self.stage.get_position()
            self.stage_x_var.set(f"{pos.x:.1f}")
            self.stage_y_var.set(f"{pos.y:.1f}")
        if self.pipette.is_connected():
            pos = self.pipette.get_position()
            self.pip_x_var.set(f"{pos.x:.1f}")
            self.pip_y_var.set(f"{pos.y:.1f}")
            self.pip_z_var.set(f"{pos.z:.1f}")

        # Status bar
        stage_txt = "N/A"
        pip_txt = "N/A"
        if self.stage.is_connected():
            p = self.stage.get_position()
            stage_txt = f"({p.x:.1f}, {p.y:.1f})"
        if self.pipette.is_connected():
            p = self.pipette.get_position()
            pip_txt = f"({p.x:.1f}, {p.y:.1f}, {p.z:.1f})"
        self.pos_label.config(text=f"Stage: {stage_txt} | Pipette: {pip_txt}")

    def _set_msg(self, msg: str):
        self.msg_label.config(text=msg)

    def _on_close(self):
        """Handle window close."""
        self._stop_preview()
        if self.collector.recording:
            if messagebox.askyesno("确认", "正在录制中，是否保存并退出？"):
                self.collector.stop_recording()
        self._disconnect_devices()
        self.root.destroy()

    def run(self):
        """Start the UI main loop."""
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MicroDreamer Data Collection UI")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    ui = CollectionUI(config_path=args.config)
    ui.collector.output_dir = Path(args.output_dir)
    ui.save_dir_var.set(args.output_dir)
    ui.run()


if __name__ == "__main__":
    main()
