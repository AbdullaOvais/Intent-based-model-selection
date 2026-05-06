# Intent-Based Model Selection for Anomaly Detection using ns-O-RAN

## Project Overview
This project studies handover misconfiguration in ns-O-RAN using real simulator telemetry and an intent-based model selection workflow. The main idea is to collect raw handover traces from ns-3 / ns-O-RAN, convert them into handover-window KPI features, train a small catalog of lightweight models, and then select the most suitable model according to the operator’s intent.

The project focuses on one exp1 handover scenario where hoSinrDifference is changed between the normal and anomaly runs. The raw traces are extracted from CellIdStats.txt, CellIdStatsHandover.txt, and DlE2PdcpStats.txt, then converted into CSV files and KPI summaries. Four models are trained on the same feature set: Random Forest, Gradient Boosting, Logistic Regression, and SVM. Their metrics are stored in a model catalog, and a rule-based selector recommends the best model based on intent such as high accuracy, low latency, or interpretability.

The reported pipeline is telemetry-driven and rule-based. Optional LLM helpers exist in the package, but they are not part of the main evaluated path.

## Setup Instructions
. Requirements
1. Python 3.10+
2. pip install -r requirements.txt
3. ns-3 / ns-O-RAN experiment environment
4. Docker for the RIC stack

## This repository is the local analysis companion for a real **ns-O-RAN handover** study.

The main idea is simple:

1. collect raw telemetry from a real ns-O-RAN / ns-3 experiment,
2. convert those raw traces into a shared KPI schema,
3. compare normal versus anomaly runs around the handover window,
4. train lightweight detectors, and
5. use a **rule-based** model selection for the given intent by operator to explain the fault and suggest handover tuning changes.



## What This Repo Contains

- `scripts/build_real_ns_o_ran_dataset.py`
  - converts raw ns-O-RAN trace bundles into the project CSV layout,
  - builds handover-window summaries,
  - and can optionally run the local diagnosis pipeline.
- `ns_oran_intent_selector/`
  - dataset helpers,
  - model training/evaluation,
  - a small model catalog,
  - and the rule-based selector.
- `README.md`
  - this guide.

The repository does **not** include the full ns-O-RAN simulator stack itself.
You run the simulator and RIC in your experiment environment, then feed the exported trace bundles into this repo.

## Prerequisites

- Python 3.10+ recommended
- `pip install -r requirements.txt`
- A real ns-O-RAN run that produces these raw files in each bundle directory:
  - `CellIdStats.txt`
  - `CellIdStatsHandover.txt`
  - `DlE2PdcpStats.txt`

## Raw Trace Bundle Layout

Each experiment bundle should look like this:

```text
normal/
  CellIdStats.txt
  CellIdStatsHandover.txt
  DlE2PdcpStats.txt

anomaly/
  CellIdStats.txt
  CellIdStatsHandover.txt
  DlE2PdcpStats.txt
```

The `normal` folder is your baseline run.
The `anomaly` folder is the misconfigured run, for example:

- lower hysteresis
- shorter time-to-trigger
- different `hoSinrDifference`

## Data Collection Workflow

The project collects data in two parts:

1. the **RIC side** in Docker, which shows the control and telemetry stack,
2. the **ns-3 side** on the host machine, which generates the raw trace files.

The raw telemetry comes from these files:

- `CellIdStats.txt`
- `CellIdStatsHandover.txt`
- `DlE2PdcpStats.txt`

These files are then copied into experiment folders and converted into CSV and KPI summaries.

### Terminal 1: Start and monitor the RIC stack

This terminal is used to make sure the RIC side is running.

```bash
docker start e2term e2mgr e2rtmansim db sample-xapp-24
docker logs -f e2term
```

live trace watch:

```bash
tail -f CellIdStatsHandover.txt

```

What this does:

- starts the Docker containers if they are stopped,
- shows E2 termination logs,
- proves that the RIC side is live.

### Terminal 2: Open the xApp container

This is optional for raw trace collection, but useful for a live demo.

```bash
docker exec -it sample-xapp-24 bash
cd /home/sample-xapp
./run_xapp.sh 2>&1 | tee /tmp/xapp_live.log
```

What this does:

- opens a shell inside the xApp container,
- lets you inspect the xApp environment,
- is useful if you want to show the RIC side interactively.

### Terminal 3: Run the ns-3 scenario for the normal case

This terminal runs the simulator and generates the raw telemetry files.

```bash
cd ~/Desktop/TWIN/ns-o-ran/ns-3-mmwave-oran
./ns3 run "scratch/scenario-zero.cc --e2TermIp=10.0.2.1 --enableE2FileLogging=1 --simTime=60 --hoSinrDifference=3 --hysteresis=3.0 --TimeToTrigger=256"
```
live trace watch:

```bash
tail -f CellIdStatsHandover.txt

```


What this does:

- runs the ns-3 / ns-O-RAN simulation,
- connects the simulator to `e2term`,
- generates the raw trace files in the simulator output folder.

For the normal run:

- `hoSinrDifference = 3`
- `Hysteresis = 3.0`
- `TimeToTrigger = 256`

### Terminal 4: Run the anomaly case

This terminal runs the same scenario again, but with one handover parameter changed.

Example 1: `hoSinrDifference` anomaly

```bash
cd ~/Desktop/TWIN/ns-o-ran/ns-3-mmwave-oran
./ns3 run "scratch/scenario-zero.cc --e2TermIp=10.0.2.1 --enableE2FileLogging=1 --simTime=60 --hoSinrDifference=18 --hysteresis=3.0 --TimeToTrigger=256"
```

Example 2: hysteresis anomaly

```bash
cd ~/Desktop/TWIN/ns-o-ran/ns-3-mmwave-oran
./ns3 run "scratch/scenario-zero.cc --e2TermIp=10.0.2.1 --enableE2FileLogging=1 --simTime=60 --hoSinrDifference=3 --hysteresis=0.5 --TimeToTrigger=256"
```

Example 3: time-to-trigger anomaly

```bash
cd ~/Desktop/TWIN/ns-o-ran/ns-3-mmwave-oran
./ns3 run "scratch/scenario-zero.cc --e2TermIp=10.0.2.1 --enableE2FileLogging=1 --simTime=60 --hoSinrDifference=3 --hysteresis=3.0 --TimeToTrigger=40"
```

What this does:

- keeps the setup the same,
- changes only one parameter,
- lets you compare normal vs anomaly cleanly.

### Terminal 5: Save the raw trace files

After each run, copy the raw files immediately, because the next run can overwrite them.

#### Normal run

```bash
mkdir -p ~/Desktop/TWIN/ns-o-ran/results/exp1/normal
cp CellIdStats.txt ~/Desktop/TWIN/ns-o-ran/results/exp1/normal/
cp CellIdStatsHandover.txt ~/Desktop/TWIN/ns-o-ran/results/exp1/normal/
cp DlE2PdcpStats.txt ~/Desktop/TWIN/ns-o-ran/results/exp1/normal/
```

#### Anomaly run

```bash
mkdir -p ~/Desktop/TWIN/ns-o-ran/results/exp1/anomaly
cp CellIdStats.txt ~/Desktop/TWIN/ns-o-ran/results/exp1/anomaly/
cp CellIdStatsHandover.txt ~/Desktop/TWIN/ns-o-ran/results/exp1/anomaly/
cp DlE2PdcpStats.txt ~/Desktop/TWIN/ns-o-ran/results/exp1/anomaly/
```

You can repeat the same folder structure for:

- `exp2` for hysteresis,
- `exp3` for time-to-trigger.



## What The Project Measures

The pipeline focuses on deep telemetry for a **single specific fault class**:

- handover threshold misconfiguration,
- hysteresis misconfiguration,
- time-to-trigger misconfiguration.

The strongest evidence comes from:

- handover event counts,
- cell transition patterns,
- packet loss around the handover window,
- received throughput around the handover window,
- delay around the handover window.


## Model Training, Catalog, and Intent Selection

After data collection, the next stage is to train a small set of candidate models on the handover-window features and store their evaluation results in a catalog.

### Trained Models

For exp1 handover misconfiguration, the current model set is:

- Random Forest
- Boosting
- Logistic Regression
- SVM

### Per-Model Training Commands

Each script trains and tests one model, then saves:

- `model.pkl`
- `metrics.json`
- `catalog_entry.json`

Example commands from the exp1 results workspace:

```bash
python3 train_exp1_random_forest.py \
  --normal-csv ../exp1_hosinr/handover_window_results/normal_handover_windows.csv \
  --anomaly-csv ../exp1_hosinr/handover_window_results/anomaly_handover_windows.csv \
  --output-dir ../exp1_hosinr/model_outputs/random_forest

python3 train_exp1_boosting.py \
  --normal-csv ../exp1_hosinr/handover_window_results/normal_handover_windows.csv \
  --anomaly-csv ../exp1_hosinr/handover_window_results/anomaly_handover_windows.csv \
  --output-dir ../exp1_hosinr/model_outputs/boosting

python3 train_exp1_logistic_regression.py \
  --normal-csv ../exp1_hosinr/handover_window_results/normal_handover_windows.csv \
  --anomaly-csv ../exp1_hosinr/handover_window_results/anomaly_handover_windows.csv \
  --output-dir ../exp1_hosinr/model_outputs/logistic_regression

python3 train_exp1_svm.py \
  --normal-csv ../exp1_hosinr/handover_window_results/normal_handover_windows.csv \
  --anomaly-csv ../exp1_hosinr/handover_window_results/anomaly_handover_windows.csv \
  --output-dir ../exp1_hosinr/model_outputs/svm
```

### Build the Final Model Catalog

After the four models finish training, combine their outputs into one catalog:

```bash
python3 scripts/build_exp1_model_catalog.py \
  --model-outputs-dir ../exp1_hosinr/model_outputs \
  --output-catalog ../exp1_hosinr/model_catalog.json \
  --output-summary ../exp1_hosinr/model_results.csv \
  --output-index ../exp1_hosinr/model_index.json
```

This creates:

- `model_catalog.json`
- `model_results.csv`
- `model_index.json`

### Rule-Based Intent Selection

The operator then provides an intent string, and the rule-based selector chooses the best model from the catalog.

Example:

```bash
python3 select_exp1_model_by_intent.py \
  --catalog "./exp/model_catalog.json" \
  --index "./exp/model_index.json" \
  --intent "Detect handover misconfiguration with high accuracy"
```
```bash
python3 select_exp1_model_by_intent.py \
  --catalog "./exp/model_catalog.json" \
  --index "./exp/model_index.json" \
  --intent "Need low latency for live inference"
```
```bash
python3 select_exp1_model_by_intent.py \
  --catalog "./exp/model_catalog.json" \
  --index "./exp/model_index.json" \
  --intent "Use a simple interpretable baseline model"
```

Typical intent mappings:

- `high accuracy` -> SVM
- `low latency` -> Boosting
- `interpretability` or `labeled` -> Random Forest
- `simple` / `baseline` / `linear` -> Logistic Regression

### What the Catalog Stores

Each catalog entry keeps:

- model name
- task type
- input type
- accuracy
- inference latency
- training cost
- label requirement
- tags
- description

## Optional Synthetic Smoke Test

If you only want a quick internal demo without real traces, the synthetic path is still available:

```bash
python3 -m ns_oran_intent_selector.pipeline
```

This is a fallback only.
The real project path is the trace-export workflow described above.

## Repository Hygiene

Keep generated files out of Git:

- `artifacts/`
- `logs/`
- `__pycache__/`
- `.DS_Store`

## Suggested GitHub Story

When you push this repo, describe it as:

> A telemetry-driven ns-O-RAN handover diagnosis workflow that converts real simulator traces into a shared KPI schema, compares normal and anomaly runs, and uses a lightweight rule-based selector to recommend handover tuning changes.








