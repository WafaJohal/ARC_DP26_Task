#!/usr/bin/env python3
"""
01_human_fitts.py — Human Fitts’ Law (teleoperation only)

Controls:
- Left-click near the end-effector to "grab" it, then drag to move.
- Release inside the target to finish a trial.
- Keyboard:
    n : next condition
    p : previous condition
    r : reset (respawn at start)
    a : analyze (plot MT vs ID for collected trials)
    s : save CSV
    q : quit
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arm_utils import Arm2D, ReachingViz, fitts_id, inside_target, make_target, plot_mt_vs_id, save_records_csv


# ---- Experiment configuration (keep consistent across all scripts) ----
TRIAL_CONDITIONS = [
    # (D, W)
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

GRAB_RADIUS = 0.15  # must click within this distance of EE to "grab" it
DEFAULT_OUT_CSV = "human_fitts_trials.csv"


class HumanFittsApp:
    def __init__(self) -> None:
        self.arm = Arm2D(l1=2.0, l2=2, elbow="down")
        self.viz = ReachingViz(self.arm, start_xy=(1.3, 0.0))

        self.records: List[Dict] = []
        self.trial_counts = {i: 0 for i in range(len(TRIAL_CONDITIONS))}
        self.current_idx = 0

        self.trial_active = False
        self.trial_start_time: float | None = None
        self.mouse_down = False
        self.grabbed = False

        self._update_target_display()

        # Connect callbacks
        c = self.viz.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("key_press_event", self.on_key)

    def current_condition(self):
        D, W = TRIAL_CONDITIONS[self.current_idx]
        target_xy = make_target(self.viz.start_xy, D, TARGET_ANGLE)
        ID = fitts_id(D, W)
        return D, W, ID, target_xy

    def _status_line(self) -> str:
        D, W, ID, _ = self.current_condition()
        n = self.trial_counts[self.current_idx]
        return f"HUMAN | Cond {self.current_idx+1}/{len(TRIAL_CONDITIONS)} | D={D:.2f}, W={W:.2f}, ID={ID:.2f} | trials={n}"

    def _update_target_display(self) -> None:
        D, W, ID, txy = self.current_condition()
        print(f"Current condition: D={D}, W={W}, ID={ID}, target={txy}")
        self.viz.set_target(txy, W)
        self.viz.set_status(self._status_line())
        self.viz.fig.canvas.draw_idle()

    def _start_trial_if_needed(self) -> None:
        if not self.trial_active:
            self.trial_active = True
            self.trial_start_time = time.time()

    def _finish_trial(self, success: bool) -> None:
        D, W, ID, txy = self.current_condition()
        mt = (time.time() - self.trial_start_time) if self.trial_start_time is not None else np.nan

        rec = dict(
            mode="human",
            condition_index=self.current_idx,
            D=float(D),
            W=float(W),
            ID=float(ID),
            movement_time_s=float(mt),
            success=bool(success),
            timestamp=time.time(),
        )
        self.records.append(rec)
        self.trial_counts[self.current_idx] += 1

        self.trial_active = False
        self.trial_start_time = None
        self.viz.set_status(self._status_line() + (" | SUCCESS" if success else " | FAIL"))
        self.viz.fig.canvas.draw_idle()

    def on_press(self, event) -> None:
        if event.inaxes != self.viz.ax:
            return
        self.mouse_down = True
        if event.xdata is None or event.ydata is None:
            return

        ee = self.arm.ee()
        click = np.array([event.xdata, event.ydata], dtype=float)
        if float(np.linalg.norm(click - ee)) <= GRAB_RADIUS:
            self.grabbed = True
            self._start_trial_if_needed()

    def on_release(self, event) -> None:
        self.mouse_down = False
        if self.grabbed:
            self.grabbed = False
            # Success if EE is inside target at release
            D, W, ID, txy = self.current_condition()
            success = inside_target(self.arm.ee(), txy, W)
            self._finish_trial(success=success)

    def on_motion(self, event) -> None:
        if event.inaxes != self.viz.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.viz.set_cursor((event.xdata, event.ydata))
        if self.grabbed:
            self.arm.set_ee(float(event.xdata), float(event.ydata))
            self.viz.update_robot()

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
        elif event.key == "a":
            if not self.records:
                self.viz.set_status(self._status_line() + " | no data yet")
                self.viz.fig.canvas.draw_idle()
                return
            df = pd.DataFrame(self.records)
            plot_mt_vs_id(df[df["success"] == True], title="Human: mean MT vs ID (successful trials)")
        elif event.key == "s":
            out = Path(DEFAULT_OUT_CSV)
            save_records_csv(self.records, str(out))
            self.viz.set_status(self._status_line() + f" | saved {out}")
            self.viz.fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(self.viz.fig)

    def run(self) -> None:
        plt.show()


if __name__ == "__main__":
    HumanFittsApp().run()
