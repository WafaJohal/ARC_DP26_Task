#!/usr/bin/env python3
"""
02_auto_fitts.py — Robot Fitts’ Law (autonomous control only)

Controls:
- Keyboard:
    space : run N trials for current condition
    n/p   : next/prev condition
    r     : reset
    a     : analyze (plot MT vs ID)
    s     : save CSV
    q     : quit

Notes:
- This uses a simple proportional controller in Cartesian space with velocity clamp.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Controller parameters
DT = 0.02
K = 2.0
VMAX = 0.8

DEFAULT_TRIALS_PER_COND = 3
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_OUT_CSV = "auto_fitts_trials.csv"


def clamp_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= vmax:
        return v
    return (v / (n + 1e-9)) * vmax


class AutoFittsApp:
    def __init__(self) -> None:
        self.arm = Arm2D(l1=2.0, l2=2.0, elbow="down")
        self.viz = ReachingViz(self.arm, start_xy=(1.3, 0.0))

        self.records: List[Dict] = []
        self.trial_counts = {i: 0 for i in range(len(TRIAL_CONDITIONS))}
        self.current_idx = 0

        # batch execution state
        self.batch_remaining = 0
        self.timeout_s = DEFAULT_TIMEOUT_S
        self.trial_start_time: Optional[float] = None
        self.trial_active = False

        self._update_target_display()

        c = self.viz.fig.canvas
        c.mpl_connect("key_press_event", self.on_key)

        self.anim = FuncAnimation(self.viz.fig, self._tick, interval=int(DT * 1000))

    def current_condition(self):
        D, W = TRIAL_CONDITIONS[self.current_idx]
        txy = make_target(self.viz.start_xy, D, TARGET_ANGLE)
        return D, W, fitts_id(D, W), txy

    def _status_line(self) -> str:
        D, W, ID, _ = self.current_condition()
        n = self.trial_counts[self.current_idx]
        return f"AUTO | Cond {self.current_idx+1}/{len(TRIAL_CONDITIONS)} | D={D:.2f}, W={W:.2f}, ID={ID:.2f} | trials={n} | batch_left={self.batch_remaining}"

    def _update_target_display(self) -> None:
        D, W, ID, txy = self.current_condition()
        self.viz.set_target(txy, W)
        self.viz.set_status(self._status_line())
        self.viz.fig.canvas.draw_idle()

    def _start_trial(self) -> None:
        self.viz.reset_robot()
        self.trial_active = True
        self.trial_start_time = time.time()

    def _finish_trial(self, success: bool) -> None:
        D, W, ID, _ = self.current_condition()
        mt = (time.time() - self.trial_start_time) if self.trial_start_time is not None else np.nan

        self.records.append(
            dict(
                mode="auto",
                condition_index=self.current_idx,
                D=float(D),
                W=float(W),
                ID=float(ID),
                movement_time_s=float(mt),
                success=bool(success),
                timeout_s=float(self.timeout_s),
                timestamp=time.time(),
            )
        )
        self.trial_counts[self.current_idx] += 1
        self.trial_active = False
        self.trial_start_time = None

        if self.batch_remaining > 0:
            self.batch_remaining -= 1

        self.viz.set_status(self._status_line() + (" | SUCCESS" if success else " | TIMEOUT"))
        self.viz.fig.canvas.draw_idle()

    def _tick(self, _frame) -> None:
        # Called by FuncAnimation
        if not self.trial_active:
            if self.batch_remaining > 0:
                self._start_trial()
                self._update_target_display()
            return

        D, W, ID, txy = self.current_condition()
        ee = self.arm.ee()
        err = (txy - ee)

        # P-controller in Cartesian space
        v = K * err
        v = clamp_norm(v, VMAX)
        ee_next = ee + v * DT

        self.arm.set_ee(float(ee_next[0]), float(ee_next[1]))
        self.viz.update_robot()

        if inside_target(self.arm.ee(), txy, W):
            self._finish_trial(success=True)
        else:
            if self.trial_start_time is not None and (time.time() - self.trial_start_time) >= self.timeout_s:
                self._finish_trial(success=False)

    def on_key(self, event) -> None:
        if event.key == " ":
            # Start a batch
            self.batch_remaining = DEFAULT_TRIALS_PER_COND
            self._update_target_display()
        elif event.key == "n":
            self.current_idx = (self.current_idx + 1) % len(TRIAL_CONDITIONS)
            self.trial_active = False
            self.batch_remaining = 0
            self.viz.reset_robot()
            self._update_target_display()
        elif event.key == "p":
            self.current_idx = (self.current_idx - 1) % len(TRIAL_CONDITIONS)
            self.trial_active = False
            self.batch_remaining = 0
            self.viz.reset_robot()
            self._update_target_display()
        elif event.key == "r":
            self.trial_active = False
            self.batch_remaining = 0
            self.viz.reset_robot()
            self.viz.set_status(self._status_line() + " | reset")
            self.viz.fig.canvas.draw_idle()
        elif event.key == "a":
            if not self.records:
                self.viz.set_status(self._status_line() + " | no data yet")
                self.viz.fig.canvas.draw_idle()
                return
            df = pd.DataFrame(self.records)
            plot_mt_vs_id(df[df["success"] == True], title="Auto: mean MT vs ID (successful trials)")
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
    AutoFittsApp().run()
