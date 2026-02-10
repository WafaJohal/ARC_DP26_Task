# 🎯 Fitts’ Law Interactive Exercises — Human, Auto, and Shared Control

This repository contains a set of interactive exercises designed to explore Fitts’ Law as a practical tool for benchmarking human and shared human-robot performance. Through three progressively structured tasks: human-only control, autonomous control, and shared control, you will experimentally measure how task difficulty, movement time, and robot assistance interact. The exercises are inspired by Pan et al. (2024) (see [paper](doc/paper.pdf)).
By reproducing simplified versions of the experiments in Python, you will both validate the classic model and investigate how autonomy changes the relationship between difficulty and performance.

## Goals

- Experience with Fitt's Law
- Collect human performance data
- Observe autonomous controllers compare to humans
- Observed simple shared-control between human and automation

## 🚀 Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run an script

```bash
python src/01_human_fitts.py
python src/02_auto_fitts.py
python src/03_shared_control.py
```

Each script opens an interactive Matplotlib window.

### Task 1 — Human control

Files:

- [`src/01_human_fitts.py`](src/01_human_fitts.py}
- [`doc/01_human_fitts.md`](doc/01_human_fitts.md)

Collect movement-time data yourself and verify Fitts’ Law.

### Task 2 — Autonomous control

Files:

- `src/02_auto_fitts.py`
- `doc/02_auto_fitts.md`

Run an automated controller and compare to human behavior.

### Task 3 — Shared control

Files:

- `src/03_shared_control.py`
- `doc/03_shared_control.md`

Blend human and automation. Adjust assistance and analyze trade-offs.

#### Bonus

Propose adaptive shared control

## Background: Fitts’ Law

MT = a + b log(D/W + 1)

Where:

- **MT** = movement time
- **D** = distance (amplitude)
- **W** = target width
- **ID** = index of difficulty

Movement time scales approximately linearly with **ID**.

## 📄 Originating paper

These exercises are inspired by:

**Pan, J., Eden, J., Oetomo, D., Johal, W., “Using Fitts’ Law to Benchmark Assisted Human-Robot Performance“, In Proceedings of the 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI 2025)”** [pdf](doc/paper.pdf)

```bibtex
@inproceedings{pan2025using,
  title={Using Fitts' Law to Benchmark Assisted Human-Robot Performance},
  author={Pan, Jiahe and Eden, Jonathan and Oetomo, Denny and Johal, Wafa},
  booktitle={2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI)},
  pages={203--212},
  year={2025},
  organization={IEEE}
}
```

## 🛠️ Repo structure

```text
.
├── requirements.txt
├── README.md
├── src
│   ├── arm_utils.py
│   ├── 01_human_fitts.py
│   ├── 02_auto_fitts.py
│   └── 03_shared_control.py
└── doc
    ├── 01_human_fitts.md
    ├── 02_auto_fitts.md
    ├── 03_shared_control.md
    └── paper.pdf
```

## 💡 Tips

- Read the paper
- Run the code; multiple trials
- Save CSV results
- Plot MT vs ID
- Compare human vs autonomous throughput
- Try different shared-control weights

Happy experimenting 🚀
