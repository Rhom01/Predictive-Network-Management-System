# Predictive Network Management System Using Machine Learning

This is a full Python implementation for a research project on predictive network management.

## What the system actually does

1. Collects real measurements from a network interface.
2. Measures throughput, traffic volume, latency, packet loss, jitter, bandwidth utilization, CPU utilization, memory utilization, interface errors/drops, and availability.
3. Stores observations in `data/network_metrics.csv`.
4. Creates forecasting targets from future observed network conditions.
5. Trains:
   - ExtraTrees classifier for future congestion/performance degradation.
   - Gradient Boosting regressors for future utilization and latency.
   - Isolation Forest for abnormal network conditions.
6. Evaluates the predictive model using time-ordered holdout data and time-series cross-validation.
7. Displays current state, historical trends, forecasts, confidence, alerts, and recommended corrective actions in Streamlit.

## Important research point

The supervised forecasting target is created from future real observations.

For each current observation, the system asks:

> Did the network enter a degraded/congested state during the next five minutes?

A positive target is created when future utilization, latency, packet loss, or jitter crosses the configured threshold.

This avoids randomly inventing traffic values for the forecasting experiment.

## Project structure

```text
predictive_network_management_system/
│
├── app.py
├── collector.py
├── train.py
├── network_utils.py
├── config.py
├── requirements.txt
│
├── data/
│   └── network_metrics.csv        # created by collector
│
├── models/
│   └── predictive_models.joblib   # created by train.py
│
└── logs/
```

## Windows installation

Open PowerShell or CMD inside this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Find your network interface

```powershell
ipconfig
```

The collector also automatically selects an active non-loopback interface.

To see interfaces from Python:

```powershell
python -c "import psutil; print(list(psutil.net_if_addrs().keys()))"
```

## Start real data collection

Recommended:

```powershell
python collector.py --probe 8.8.8.8
```

Or use your gateway:

```powershell
python collector.py
```

For a specific interface:

```powershell
python collector.py --interface "Wi-Fi" --probe 8.8.8.8
```

Leave the collector running.

## Collect enough data

The model is forecasting five minutes into the future.

Do not train immediately after starting the collector.

For a credible academic experiment, collect a sufficiently long time series containing:

- normal traffic;
- busy periods;
- application traffic;
- latency changes;
- packet-loss events;
- different utilization levels.

At minimum, the code requires 300 usable observations. More data is strongly preferable for a serious study.

## Train the machine-learning models

After collecting enough measurements:

```powershell
python train.py
```

This creates:

```text
models/predictive_models.joblib
data/model_evaluation.json
```

## Start the dashboard

Open another terminal:

```powershell
.venv\Scripts\activate
streamlit run app.py
```

## Model design

### 1. Future congestion classifier

ExtraTreesClassifier predicts:

```text
0 = likely normal
1 = likely degraded/congested
```

The prediction horizon is five minutes.

Features include:

- bandwidth utilization;
- throughput;
- traffic volume;
- packet rate;
- latency;
- packet loss;
- jitter;
- CPU utilization;
- memory utilization;
- interface errors;
- interface drops;
- network availability;
- rolling means;
- rolling standard deviations;
- short-term deltas.

### 2. Performance forecasting

GradientBoostingRegressor predicts future maximum:

- bandwidth utilization;
- latency.

MAE is reported.

### 3. Abnormal-condition detection

Isolation Forest is trained on normal observations and flags statistically unusual current network states.

## Why this is not the same as a simple anomaly detector

The core research problem is forecasting.

The classifier uses current and recent network observations to predict a condition in a future window. The dashboard therefore has a genuine predictive component rather than only saying that the current traffic is abnormal.

## Evaluation

The code uses chronological splitting rather than random train/test shuffling.

This matters because random shuffling can leak future information into the training set in a time-series forecasting problem.

Reported metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable
- MAE for regression
- Time-series cross-validation results

## Conventional reactive baseline

For the thesis, define the conventional approach as:

> An alert is raised only after a threshold is crossed in the current observation.

The predictive system is:

> An alert is raised before the threshold is crossed when the ML model predicts a future degraded state.

Compare the two using:

- detection lead time;
- false alarm rate;
- missed-event rate;
- packet loss;
- throughput;
- latency;
- availability;
- resource utilization.

The dashboard already provides the measurements required for this comparison.

## Network measurement note

Latency, jitter, packet loss, and availability are measured by active probing of the configured host.

Bandwidth utilization is calculated from interface byte counters and the interface speed.

If the operating system cannot report interface speed, `DEFAULT_INTERFACE_SPEED_MBPS` in `config.py` is used as a fallback. For a thesis experiment, set that value to the actual link capacity when necessary.

## Dataset / benchmark integration

Your earlier UNSW-NB15 work can remain a separate benchmark component for abnormal-traffic detection. UNSW-NB15 is useful for evaluating network-traffic anomaly classification, but it does not directly provide all of the live performance variables required by this predictive network-management study.

For this project, the live time-series collected by `collector.py` should be the primary forecasting dataset.

Do not describe the previously generated Excel anomaly workbook as real-world ground truth. It can be used for software testing, but it should not be presented as a real network measurement source in the thesis.

## Recommended thesis architecture

```text
REAL NETWORK
     │
     ▼
Network Interface + Active Probe
     │
     ▼
Data Collection Layer
     │
     ├── throughput
     ├── bandwidth utilization
     ├── latency
     ├── packet loss
     ├── jitter
     ├── traffic volume
     ├── CPU utilization
     ├── memory utilization
     └── availability
     │
     ▼
Time-Series Storage
     │
     ▼
Preprocessing + Feature Engineering
     │
     ├── rolling statistics
     ├── trends/deltas
     └── missing-value handling
     │
     ├───────────────┬─────────────────┐
     ▼               ▼                 ▼
Congestion       Performance        Anomaly
Classifier       Regression         Detector
     │               │                 │
     └───────────────┴─────────────────┘
                     │
                     ▼
             Prediction Engine
                     │
                     ▼
             Decision/Alert Layer
                     │
                     ▼
               Streamlit Dashboard
```
