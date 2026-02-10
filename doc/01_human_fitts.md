# Exercise 1 — Human Fitts’ Law (Teleoperation Only)

## How to run

1. Install deps: `pip install -r requirements.txt`
2. Run: `python 01_human_fitts.py`

## Controls
- Left-click near the end-effector to grab, drag to move, release to attempt the target.
- Keyboard: `n` next condition, `p` previous, `r` reset, `a` analyze plot, `s` save CSV, `q` quit.

---

# Notebook 1 — Human Fitts’ Law (Teleoperation Only)

**Goal:** Understand Fitts’ Law through **human-controlled** reaching.

**Objectives**
1. Understand Fitts’ Law and Index of Difficulty (ID)
2. Measure Movement Time (MT) in reaching tasks
3. Collect experimental human data


## 0. Conceptual introduction

We model reaching as moving the end-effector (EE) from a **start** position to a **circular target**.

- **D**: distance from start to target center  
- **W**: target width (diameter of the target)  
- **ID**: Index of Difficulty  
\[
ID = \log_2\left(\frac{D}{W}+1\right)
\]

You will measure:
- **MT**: time from trial start until the EE first enters the target (or timeout)
- **Success**: reached before timeout


## 1. Environment & robot visualization
Run the next cells to define the arm and visualization.

## 2. Teleoperation experiment

**Controls**
- Click **Respawn** to reset the robot to the start pose (**required before each new reaching task**).
- Click **Start Trial** to begin timing.
- Click-and-drag **near the EE** to move it (mouse teleoperation).
- Reach the target as quickly and accurately as you can.

**Experimental rigor prompts**
- Run multiple trials per condition (e.g., 3–10).
- Keep posture / mouse sensitivity consistent.
- If something goes wrong, mark fail and redo.

## Student tasks 

1. Run ≥3 trials per condition (more is better).
2. Export your CSV.
3. Create ID vs MT plots **outside this notebook**.
4. Anaylise the results
5. Reflect: how did W and D change difficulty?
6. Report on results and your reflection
