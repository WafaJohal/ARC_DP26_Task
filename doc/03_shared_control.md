# Exercise 3 — Shared Control & Blending Policy

## How to run

1. Install deps: `pip install -r requirements.txt`
2. Run: `python 03_shared_control.py`

## Controls
- Hold left mouse button and move cursor to command the robot.
- Keyboard: `n/p` next/prev condition, `r` reset, `t` toggle policy (fixed/adaptive), `a` analyze plot, `s` save CSV, `q` quit.
- Slider: `alpha` (used in fixed policy).

---

# Notebook 3 — Shared Control & Blending Policy

**Goal:** Design and evaluate shared control strategies.

You will test:
- **Fixed blending** \(u = \alpha u_{auto} + (1-\alpha) u_{human}\)
- **Adaptive blending** \(\alpha(W, D)\) (student task)



