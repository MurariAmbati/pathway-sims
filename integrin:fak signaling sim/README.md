integrin / fak signaling (integrin_fak)

cell adhesion, migration, mechanotransduction, and ecm coupling via a small, tunable ode model.

goal
- interactively explore how integrin activation drives fak phosphorylation and downstream adhesion signaling
- includes a compact extension (talin/kindlin, src, erk) while staying small
- visualize trajectories, fluxes, parameter sweeps, and a live “process” view

run

quickstart
- from this folder: `streamlit run streamlit_app.py`
- open: `http://localhost:8501`

clean install (recommended)
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `streamlit run streamlit_app.py`

troubleshooting
- if you see numpy / binary-compat errors, make sure you have `numpy<2` (see requirements.txt)
- if port 8501 is busy: `streamlit run streamlit_app.py --server.port 8502`

notes
- this is a compact phenomenological model (not a full mechanochemical + spatial adhesome)
- units are arbitrary; interpret trends/relative changes, not absolute concentrations
