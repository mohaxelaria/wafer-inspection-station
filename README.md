# Wafer Inspection Station

A working simulation of a semiconductor **optical inspection tool** and the
operator software that drives it: machine control, 10 Hz telemetry, live wafer
mapping, automatic defect-pattern classification, alarms and result storage.

The point of the project is the part that is hard to show on a CV: not a model
in a notebook, but a model inside a running machine, with an operator screen in
front of it and a database behind it.

![status](https://img.shields.io/badge/milestone-2%20of%204-blue) ![macro F1](https://img.shields.io/badge/WM--811K%20macro%20F1-0.873-brightgreen)

## What it does

- **Simulated tool** - a cassette of wafers, a stage that walks every die,
  illumination that degrades, and faults (stage timeout, wafer misalignment)
  that put the machine into a real fault state and need operator action.
- **Operator console (HMI)** - live wafer map filling in die by die, stage
  position, yield and throughput tiles, a throughput chart, an alarm panel with
  severity, and machine controls.
- **Automatic Defect Classification** - every finished wafer is classified into
  a WM-811K pattern class (`none`, `center`, `donut`, `edge_ring`, `scratch`,
  `loc`) and scored against the simulator's ground truth.
- **Result store** - lots, wafers, yields, predictions and alarms in SQL, with a
  running detector-accuracy metric.

## Architecture

```
  machine.py            main.py                     index.html
 +------------+      +-------------------+        +----------------+
 | simulated  | ---> | FastAPI           | --WS-> | Vue 3 operator |
 | inspection |      |  - control API    |        | console        |
 | tool 10Hz  | <--- |  - event fan-out  | <-REST-|                |
 +------------+      +---------+---------+        +----------------+
       |                       |
       v                       v
  detector.py              db.py (SQL)
  ADC classifier           lots / wafers / alarms
```

Every layer talks through a narrow interface, so each one can be replaced
independently: the heuristic detector for a trained CNN, SQLite for PostgreSQL,
the simulator for a real tool.

## Run it

With Anaconda (recommended - `run.bat` does all of this for you):

```bash
conda env create -f environment.yml
conda activate wafer-inspection
uvicorn backend.main:app --reload
```

Or into an existing environment, with plain pip:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000>, press **Start**.

If the project sits on a network or shared drive where SQLite cannot lock
files, point the database somewhere local: `set WIS_DB_PATH=C:\temp\inspection.db`.

## Score the detector

```bash
python -m scripts.bench_detector 100
```

```
detector: heuristic-v1   wafers: 600   accuracy: 97.7%

                  none     center      donut  edge_ring    scratch        loc
none               100          0          0          0          0          0   recall 100%
center               0        100          0          0          0          0   recall 100%
donut                0          0        100          0          0          0   recall 100%
edge_ring            0          0          0        100          0          0   recall 100%
scratch              3          0          0          0         97          0   recall  97%
loc                  3          2          0          0          6         89   recall  89%
```

That number is honest but flattering: the simulator draws clean parametric
patterns, and hand-written radial features are very good at clean parametric
patterns. Real wafer maps are noisy, mixed-mode and unbalanced, which is
exactly where the heuristic falls over and a CNN earns its place - measuring
that gap is milestone 2.

## Milestone 2 - train the real detector

The heuristic never sees real data. To replace it:

```bash
pip install -r requirements-ml.txt          # install PyTorch from pytorch.org first
kaggle datasets download -d qingyi/wm811k-wafer-map -p data --unzip
python -m ml.data --size 32 --max-per-class 20000   # build the cached tensors
python -m ml.train --epochs 20                       # writes ml/checkpoints/wafer_cnn.pt
python -m ml.evaluate                                # heuristic vs CNN -> ml/RESULTS.md
```

Restart the server and the console header switches from `heuristic-v1` to the
CNN automatically - `backend/detector_cnn.py` loads the checkpoint if it
exists and falls back to the heuristic if it does not.

Two things about WM-811K that shape the code: only ~173k of the 811k wafers
carry a failure label, and about 85% of those are `none`. The cache subsamples
the majority class and the training loss is class-weighted; without both, the
model reaches high accuracy by predicting `none` for everything, which is why
`ml/evaluate.py` reports **macro F1** and not only accuracy.

## Results: what the model actually bought

Both detectors scored on the same held-out split of **6,834 real wafer maps**,
with identical input. Full per-class tables in [ml/RESULTS.md](ml/RESULTS.md).

| detector | accuracy | macro F1 |
|---|---:|---:|
| Heuristic, on **simulated** wafers | 0.977 | - |
| Heuristic, on **real** WM-811K | 0.476 | 0.235 |
| CNN trained on WM-811K | **0.923** | **0.873** |

The first two rows are the same code. Hand-written radial features score 97.7%
on the clean parametric patterns my simulator draws and 47.6% on real wafers -
that collapse is the honest reason to train a model, and it is why the
simulator keeps ground truth in the first place.

Where the heuristic fails is specific, not general:

- `edge_loc`, `random`, `near_full` - **F1 = 0.00**. It has no rule for them;
  a rule-based system can only find shapes somebody wrote a rule for.
- `center` - recall **0.04**. The rule wants most failures inside r < 0.35.
  Real centre defects are larger, noisier and off-centre, so the threshold
  almost never fires.
- `edge_ring` - recall 0.96 but F1 only 0.57: it labels far too many wafers
  `edge_ring`, because real wafers have failing dies near the edge for reasons
  that have nothing to do with a ring defect.

The CNN's own weak spot is `scratch` (F1 0.70) and, behind it, `loc` (0.80).
Both are consistent with the preprocessing rather than the model: a scratch is
often one or two dies wide, and nearest-neighbour downsampling to 32x32 breaks
a thin line into disconnected dots. Raising the input resolution is the next
experiment, not a bigger network.

**Metric note.** 66% of the labelled wafers in this dataset are `none`, so a
model that predicts `none` for everything already scores about 0.66 accuracy.
Every number here is reported with macro F1 beside it for that reason.

## Roadmap

| Milestone | Content |
|---|---|
| 1 - done | Simulator, FastAPI + WebSocket, Vue console, SQLite store, heuristic ADC |
| **2 - done** | CNN trained on the real WM-811K dataset behind the same `Detector` interface; heuristic vs CNN on real data |
| 3 | PostgreSQL + Redis, Docker Compose, one-command startup |
| 4 | SECS/GEM equipment interface (SEMI E5/E30) and SPC control charts with Western Electric rules |

## Layout

```
backend/
  wafer.py      wafer geometry, WM-811K-style defect pattern generation
  machine.py    the simulated tool: states, telemetry, faults, scan loop
  detector.py   Detector interface + heuristic baseline classifier
  db.py         SQL result store
  main.py       FastAPI app: control API, WebSocket fan-out, static hosting
frontend/
  index.html    Vue 3 operator console (canvas wafer map + charts)
scripts/
  bench_detector.py   offline accuracy + confusion matrix on simulated wafers
ml/
  labels.py     shared 9-class vocabulary (WM-811K names -> snake_case)
  data.py       WM-811K loader, resize, class balancing, stratified split
  model.py      small CNN (~290k parameters)
  train.py      class-weighted training loop, checkpointing, per-class F1
  compat.py     real wafer map -> Wafer object, so the heuristic can be scored on it
  evaluate.py   head-to-head heuristic vs CNN -> RESULTS.md
```

## Notes

Wafer coordinates are normalised to a unit circle, dies are scanned in
serpentine order like a real stage, and the die grid, scan rate and lot size are
all parameters - so the same code drives a 26x26 demo wafer or a realistic map.
