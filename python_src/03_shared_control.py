#!/usr/bin/env python3
"""
03_shared_control.py — Shared Control & Blending Policy

Idea:
- Human provides an intended direction by moving the mouse cursor (while holding left mouse button).
- Autonomous controller drives toward the target.
- A blending factor alpha ∈ [0,1] mixes commands: v = alpha * v_human + (1-alpha) * v_auto

Controls:
- Hold left mouse button and move cursor to command the robot.
- Keyboard:
    n/p   : next/prev condition
    r     : reset
    t     : toggle policy (fixed <-> adaptive)
    a     : analyze (plot MT vs ID)
    s     : save CSV
    q     : quit

UI:
- Alpha slider is used when policy == fixed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from arm_utils import Arm2D, ReachingViz, fitts_id, inside_target, make_target, plot_mt_vs_id, save_records_csv


TRIAL_CONDITIONS = [
     (0.6, 0.20),
    (0.9, 0.20),
    (1.2, 0.20),
    (0.9, 0.12),
    (1.2, 0.12),
    (1.2, 0.08),
    (2.2, 0.12),
    (2.2, 0.08),
    (0.6, 0.20),
    (0.9, 0.20),
    (1.2, 0.20),
    (2.2, 0.4),
]
TARGET_ANGLE = 0.0

DT = 0.02
K_H = 3.0
K_A = 2.0
VMAX = 0.9
DEFAULT_TIMEOUT_S = 12.0
DEFAULT_OUT_CSV = "shared_control_trials.csv"


def clamp_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= vmax:
        return v
    return (v / (n + 1e-9)) * vmax


class SharedControlApp:
    def __init__(self) -> None:
        self.arm = Arm2D(l1=2.0, l2=2.0, elbow="down")
        self.viz = ReachingViz(self.arm, start_xy=(1.3, 0.0))

        self.records: List[Dict] = []
        self.trial_counts = {i: 0 for i in range(len(TRIAL_CONDITIONS))}
        self.current_idx = 0

        self.trial_active = False
        self.trial_start_time: Optional[float] = None
        self.timeout_s = DEFAULT_TIMEOUT_S

        self.mouse_down = False
        self.last_alpha_used = 0.0

        self.policy = "fixed"  # or "adaptive"

        # Alpha slider (fixed policy)
        self._init_slider()

        self._update_target_display()

        c = self.viz.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("key_press_event", self.on_key)

        self.anim = FuncAnimation(self.viz.fig, self._tick, interval=int(DT * 1000))

    def _init_slider(self) -> None:
        # Make room at the bottom for the slider
        self.viz.fig.subplots_adjust(bottom=0.18)
        ax_slider = self.viz.fig.add_axes([0.15, 0.06, 0.7, 0.04])
        self.alpha_slider = Slider(ax_slider, "alpha", valmin=0.0, valmax=1.0, valinit=0.5)
        self.alpha_slider.valtext.set_text("0.50")

    def current_condition(self):
        D, W = TRIAL_CONDITIONS[self.current_idx]
        txy = make_target(self.viz.start_xy, D, TARGET_ANGLE)
        return D, W, fitts_id(D, W), txy

    def _status_line(self) -> str:
        D, W, ID, _ = self.current_condition()
        n = self.trial_counts[self.current_idx]
        return f"SHARED({self.policy}) | Cond {self.current_idx+1}/{len(TRIAL_CONDITIONS)} | D={D:.2f}, W={W:.2f}, ID={ID:.2f} | trials={n} | alpha_used={self.last_alpha_used:.2f}"

    def _update_target_display(self) -> None:
        D, W, ID, txy = self.current_condition()
        self.viz.set_target(txy, W)
        self.viz.set_status(self._status_line())
        self.viz.fig.canvas.draw_idle()

    # -------------------------
    # Policy (students edit)
    # -------------------------

    def adaptive_alpha(self, ee: np.ndarray, cursor: Optional[np.ndarray], target: np.ndarray, W: float) -> float:
        """
        Placeholder adaptive blending policy.

        Return alpha in [0,1] where:
            alpha=1  -> pure human
            alpha=0  -> pure autonomy

        Students: redesign this to be meaningful.
        Example signals you might use:
        - distance(cursor, target) vs distance(ee, target)
        - angle between human intent and autonomy direction
        - whether the cursor is inside/near the target
        - target width W (harder tasks may benefit from more autonomy)
        """
        if cursor is None:
            return 0.0  # no human input -> rely on autonomy

        # Simple heuristic: if human cursor is close to target, trust human more.
        d_ct = float(np.linalg.norm(cursor - target))
        d0 = max(1e-6, W) * 2.0
        alpha = 1.0 - (d_ct / (d_ct + d0))
        return float(np.clip(alpha, 0.0, 1.0))

    # -------------------------
    # Trial bookkeeping
    # -------------------------

    def _start_trial_if_needed(self) -> None:
        if not self.trial_active:
            self.trial_active = True
            self.trial_start_time = time.time()

    def _finish_trial(self, success: bool) -> None:
        D, W, ID, _ = self.current_condition()
        mt = (time.time() - self.trial_start_time) if self.trial_start_time is not None else np.nan

        self.records.append(
            dict(
                mode="shared",
                policy=self.policy,
                condition_index=self.current_idx,
                D=float(D),
                W=float(W),
                ID=float(ID),
                alpha_used=float(self.last_alpha_used),
                movement_time_s=float(mt),
                success=bool(success),
                timeout_s=float(self.timeout_s),
                timestamp=time.time(),
            )
        )
        self.trial_counts[self.current_idx] += 1

        self.trial_active = False
        self.trial_start_time = None

        self.viz.set_status(self._status_line() + (" | SUCCESS" if success else " | TIMEOUT"))
        self.viz.fig.canvas.draw_idle()

    # -------------------------
    # Event handlers
    # -------------------------

    def on_press(self, event) -> None:
        if event.inaxes != self.viz.ax:
            return
        if event.button == 1:
            self.mouse_down = True
            self._start_trial_if_needed()

    def on_release(self, event) -> None:
        if event.button == 1:
            self.mouse_down = False

    def on_motion(self, event) -> None:
        if event.inaxes != self.viz.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.viz.set_cursor((event.xdata, event.ydata))

    def on_key(self, event) -> None:
        if event.key == "n":
            self.current_idx = (self.current_idx + 1) % len(TRIAL_CONDITIONS)
            self.viz.reset_robot()
            self.trial_active = False
            self._update_target_display()
        elif event.key == "p":
            self.current_idx = (self.current_idx - 1) % len(TRIAL_CONDITIONS)
            self.viz.reset_robot()
            self.trial_active = False
            self._update_target_display()
        elif event.key == "r":
            self.viz.reset_robot()
            self.trial_active = False
            self.viz.set_status(self._status_line() + " | reset")
            self.viz.fig.canvas.draw_idle()
        elif event.key == "t":
            self.policy = "adaptive" if self.policy == "fixed" else "fixed"
            self.viz.set_status(self._status_line() + " | toggled policy")
            self.viz.fig.canvas.draw_idle()
        elif event.key == "a":
            if not self.records:
                self.viz.set_status(self._status_line() + " | no data yet")
                self.viz.fig.canvas.draw_idle()
                return
            df = pd.DataFrame(self.records)
            plot_mt_vs_id(df[df["success"] == True], title="Shared control: mean MT vs ID (successful trials)")
        elif event.key == "s":
            out = Path(DEFAULT_OUT_CSV)
            save_records_csv(self.records, str(out))
            self.viz.set_status(self._status_line() + f" | saved {out}")
            self.viz.fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(self.viz.fig)

    # -------------------------
    # Control loop
    # -------------------------

    def _tick(self, _frame) -> None:
        D, W, ID, target = self.current_condition()

        ee = self.arm.ee()

        # Autonomy command
        v_a = K_A * (target - ee)

        # Human command (only when mouse is held and cursor exists)
        cursor = self.viz.cursor_xy
        if self.mouse_down and cursor is not None:
            v_h = K_H * (cursor - ee)
        else:
            v_h = np.zeros(2, dtype=float)

        # Determine alpha
        if self.policy == "fixed":
            alpha = float(self.alpha_slider.val)
        else:
            alpha = float(self.adaptive_alpha(ee=ee, cursor=cursor if self.mouse_down else None, target=target, W=W))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.last_alpha_used = alpha

        v = alpha * v_h + (1.0 - alpha) * v_a
        v = clamp_norm(v, VMAX)

        ee_next = ee + v * DT
        self.arm.set_ee(float(ee_next[0]), float(ee_next[1]))
        self.viz.update_robot()

        # Trial completion logic
        if self.trial_active:
            if inside_target(self.arm.ee(), target, W):
                self._finish_trial(success=True)
            else:
                if self.trial_start_time is not None and (time.time() - self.trial_start_time) >= self.timeout_s:
                    self._finish_trial(success=False)

        self.viz.set_status(self._status_line())

    def run(self) -> None:
        plt.show()


if __name__ == "__main__":
    SharedControlApp().run()
