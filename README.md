# 🎯 Fitts’ Law Interactive Exercises — Human, Auto, and Shared Control

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
python 01_human_fitts.py
python 02_auto_fitts.py
python 03_shared_control.py
```

Each script opens an interactive Matplotlib window.

### Task 1 — Human control

Files:

- 01_human_fitts.py
- 01_human_fitts.md

Collect movement-time data yourself and verify Fitts’ Law.

---

### Task 2 — Autonomous control

Files:

- 02_auto_fitts.py
- 02_auto_fitts.md

Run an automated controller and compare to human behavior.

---

### Task 3 — Shared control

Files:

- 03_shared_control.py
- 03_shared_control.md

Blend human and automation. Adjust assistance and analyze trade-offs.

#### Bonus

Propose adaptive shared control

---

## 🧠 Background: Fitts’ Law

MT = a + b log(D/W + 1)

Where:

- MT = movement time
- D = distance
- W = target width
- ID = index of difficulty

Movement time scales approximately linearly with ID.

---

## 📄 Notes on the original paper

Paul Fitts (1954) framed pointing as an information channel:

- Speed–accuracy tradeoff
- Logarithmic difficulty scaling
- Linear MT vs ID
- Throughput measured in bits/second

Your collected data should reproduce this relationship.

---

## 🛠️ Repo structure

```bash
.
├── requirements.txt
├──| arm_utils.py
├── 
├── 01_human_fitts.py
├── 01_human_fitts.md
├── 02_auto_fitts.py
├── 02_auto_fitts.md
├── 03_shared_control.py
└── 03_shared_control.md
```

---

## 💡 Tips

- Run multiple trials
- Save CSV results
- Plot MT vs ID
- Compare human vs autonomous throughput
- Try different shared-control weights

Happy experimenting 🚀
