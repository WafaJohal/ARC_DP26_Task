"""
Shared utilities for the 2-DOF planar arm Fitts' Law & shared control exercises.

These scripts are designed to run as *normal Python programs* (no Jupyter required).
They use Matplotlib's built-in event handling + widgets for interactivity.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


####
# Issues: add keynoard control
# fix target respawn 


# ----------------------------
# Arm model (2-link planar)
# ----------------------------

class Arm2D:
    """
    2-link planar arm with geometric inverse kinematics (IK).
    Control interface: set desired end-effector (x, y) in Cartesian space.
    """

    def __init__(self, l1: float = 2.0, l2: float = 2.0, elbow: str = "down") -> None:
        self.l1 = float(l1)
        self.l2 = float(l2)
        if elbow not in ("down", "up"):
            raise ValueError("elbow must be 'down' or 'up'")
        self.elbow = elbow
        self.q = np.array([0.0, 0.0], dtype=float)

    def fk(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward kinematics: returns points [[0,0], [x1,y1], [x2,y2]]."""
        if q is None:
            q = self.q
        q1, q2 = float(q[0]), float(q[1])
        x1 = self.l1 * math.cos(q1)
        y1 = self.l1 * math.sin(q1)
        x2 = x1 + self.l2 * math.cos(q1 + q2)
        y2 = y1 + self.l2 * math.sin(q1 + q2)
        return np.array([[0.0, 0.0], [x1, y1], [x2, y2]], dtype=float)

    def ee(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """End-effector position."""
        return self.fk(q)[-1]

    def ik(self, x: float, y: float) -> np.ndarray:
        """Geometric IK with elbow-up/down selection."""
        l1, l2 = self.l1, self.l2
        r2 = x * x + y * y
        c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2_pos = math.sqrt(max(0.0, 1.0 - c2 * c2))
        s2 = -s2_pos if self.elbow == "down" else s2_pos
        q2 = math.atan2(s2, c2)
        k1 = l1 + l2 * c2
        k2 = l2 * s2
        q1 = math.atan2(y, x) - math.atan2(k2, k1)
        return np.array([q1, q2], dtype=float)

    def set_ee(self, x: float, y: float) -> None:
        """Set end-effector with reach clamping to the reachable annulus."""
        r = math.sqrt(x * x + y * y)
        r_min = abs(self.l1 - self.l2) + 1e-6
        r_max = (self.l1 + self.l2) - 1e-6
        if r < r_min:
            s = r_min / (r + 1e-9)
            x, y = x * s, y * s
        elif r > r_max:
            s = r_max / (r + 1e-9)
            x, y = x * s, y * s
        self.q = self.ik(float(x), float(y))

    def reset_to_start(self, start_xy: Tuple[float, float]) -> None:
        self.set_ee(float(start_xy[0]), float(start_xy[1]))


# ----------------------------
# Viz helpers
# ----------------------------

class ReachingViz:
    """Matplotlib visualization + cursor overlay."""

    def __init__(self, arm: Arm2D, start_xy: Tuple[float, float] = (1.3, 0.0)) -> None:
        self.arm = arm
        self.start_xy = np.array(start_xy, dtype=float)
        self.arm.reset_to_start(tuple(self.start_xy))

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_aspect("equal", "box")
        self.ax.grid(True, alpha=0.3)
        #self.ax.set_legend("Keyboard controls: [n] next condition, [p] previous, [r] reset, [a] analyze plot, [s] save CSV, [q] quit.")

        reach = arm.l1 + arm.l2
        self.ax.set_xlim(-1, reach + 0.2)
        self.ax.set_ylim(-1, reach + 0.2)

        pts = self.arm.fk()
        (self.link_line,) = self.ax.plot(pts[:, 0], pts[:, 1], marker="o", linewidth=3)
        self.ee_sc = self.ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=80)
        self.start_sc = self.ax.scatter([self.start_xy[0]], [self.start_xy[1]], s=80, marker="s")

        self.target_patch = Circle((0, 0), radius=0.05, fill=False, linewidth=3)
        self.ax.add_patch(self.target_patch)

        self.cursor_sc = self.ax.scatter([], [], s=60, marker="x")
        self.cursor_xy: Optional[np.ndarray] = None

        self.status = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left")

        self.fig.canvas.draw_idle()

    def set_target(self, xy: np.ndarray, W: float) -> None:
        self.target_patch.center = (float(xy[0]), float(xy[1]))
        self.target_patch.radius = float(W) / 2.0

    def set_status(self, msg: str) -> None:
        self.status.set_text(msg)

    def update_robot(self) -> None:
        pts = self.arm.fk()
        self.link_line.set_data(pts[:, 0], pts[:, 1])
        self.ee_sc.set_offsets([pts[-1]])
        self.fig.canvas.draw_idle()

    def set_cursor(self, xy: Optional[Tuple[float, float]]) -> None:
        if xy is None:
            self.cursor_xy = None
            self.cursor_sc.set_offsets(np.empty((0, 2)))
        else:
            self.cursor_xy = np.array([float(xy[0]), float(xy[1])], dtype=float)
            self.cursor_sc.set_offsets([self.cursor_xy])
        self.fig.canvas.draw_idle()

    def reset_robot(self) -> None:
        self.arm.reset_to_start(tuple(self.start_xy))
        self.update_robot()
        self.set_cursor(None)


# ----------------------------
# Fitts helpers
# ----------------------------

def fitts_id(D: float, W: float) -> float:
    """Shannon formulation: ID = log2(D/W + 1)."""
    return float(np.log2(D / W + 1.0))


def make_target(start_xy: np.ndarray, D: float, angle: float = 0.0) -> np.ndarray:
    target = np.array([start_xy[0] + D * math.cos(angle), start_xy[1] + D * math.sin(angle)], dtype=float)
    #print(f"Generated target at {target} for D={D}, angle={angle}")
    return target


def inside_target(ee_xy: np.ndarray, target_xy: np.ndarray, W: float) -> bool:
    return float(np.linalg.norm(ee_xy - target_xy)) <= (float(W) / 2.0)


def save_records_csv(records: List[Dict], out_csv: str) -> None:
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)


def plot_mt_vs_id(df: pd.DataFrame, title: str) -> None:
    # Aggregate by condition/ID
    if "ID" not in df.columns or "movement_time_s" not in df.columns:
        print("Dataframe missing required columns for plotting.")
        return
    g = df.groupby("ID", as_index=False)["movement_time_s"].mean().sort_values("ID")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["ID"].values, g["movement_time_s"].values, marker="o")
    ax.set_xlabel("Index of Difficulty (ID)")
    ax.set_ylabel("Mean movement time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
