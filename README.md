# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. We match noisy GPS tracks to the road network. A candidate classifier (LightGBM, Bayesian-tuned) scores the nearby arcs, then an HMM decoded with Viterbi picks a connected route. It is served by a FastAPI backend (laptop 1) and a Streamlit UI (laptop 2).

## Setup
```bash
cd pathsnap
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name pathsnap --display-name "PathSnap (.venv)"
```
The raw data is read in place from `../Student folder/` (`train/`, `student_test/`) and never modified.

## Layout
| Path | Contents |
|---|---|
| `src/data.py` | Raw file loaders, haversine, local metric projection |
| `src/features.py` | Candidate generation, route-to-GPS label alignment (DP), 19 candidate features |
| `src/models.py`, `src/metrics.py` | Model zoo (6 families); metrics (macro F1, log loss, Top-1, route continuity, ...) |
| `src/hmm.py` | HMM + Viterbi decoder over the directed road graph |
| `src/pipeline.py` | End-to-end inference for one trip, plus server-side scoring |
| `notebooks/01_eda.ipynb` | Step 1: dataset exploration and data-quality log |
| `notebooks/02_candidates_features.ipynb` | Step 2: labels, radius/K selection, feature tables |
| `notebooks/03_models.ipynb` | Step 3: 6 models, GroupKFold CV, student_test comparison, plots |
| `notebooks/04_bayes_opt.ipynb` | Step 3b: Optuna (TPE) tuning of the selected model, tuned vs untuned |
| `notebooks/05_hmm.ipynb` | Step 4: decoder tuning, classifier-only vs HMM, Folium maps |
| `api/main.py` | FastAPI backend |
| `ui/app.py` | Streamlit evaluator UI |
| `artifacts/` | `model.joblib`, `model_config.json`, `hmm_config.json` |
| `outputs/` | CSV results, plots, Folium maps for every step |
| `tests/` | Self-checks: `.venv/bin/python tests/test_features.py && .venv/bin/python tests/test_hmm.py` |

## Reproduce
```bash
cd notebooks
for nb in 01_eda 02_candidates_features 03_models 04_bayes_opt 05_hmm; do
  ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $nb.ipynb
done
```

## Deployment (two laptops on the same Wi-Fi)
**Laptop 1: API.** It needs this repo, the venv, `artifacts/` and the evaluation trip folder.
```bash
cd pathsnap
PATHSNAP_DATA_DIR="/path/to/evaluation_trips" .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0        # macOS: laptop 1's IP address, e.g. 192.168.1.23
```
`PATHSNAP_DATA_DIR` must contain one folder per trip (`<id>/<id>.track|.nodes|.arcs|.route`). The `.route` file is read only after prediction, to compute accuracy on the server. It is never sent to the model or the UI. Every request is logged in this terminal.

Endpoints: `GET /health`, `GET /trips`, `GET /predict/{trip_id}`.

**Laptop 2: UI.**
```bash
pip install -r ui/requirements.txt
PATHSNAP_API="http://<laptop-1-ip>:8000" streamlit run ui/app.py
```
Choose a trip, then click **Run map-matching**. The UI shows the predicted route on a map and the accuracy numbers the server computed.
