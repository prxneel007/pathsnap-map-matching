# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. We match noisy GPS tracks to the road network: a candidate classifier scores nearby road arcs, then an HMM with Viterbi decoding picks a connected route.

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
| `notebooks/01_eda.ipynb` | Step 1: dataset exploration and data-quality log |
| `notebooks/02_candidates_features.ipynb` | Step 2: labels, radius/K selection, feature tables |
| `outputs/eda/`, `outputs/features/` | CSV results, plots, Folium maps |
| `tests/test_features.py` | Self-check: `.venv/bin/python tests/test_features.py` |

## Reproduce
```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_candidates_features.ipynb
```
