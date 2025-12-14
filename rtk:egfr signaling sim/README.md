rtk / egfr signaling (rtk_egfr)

growth factor signaling via receptor tyrosine kinases, cell proliferation and survival.

what this is
- a small but feature-rich egfr/rtk signaling sandbox: ligand → egfr → ras/raf/mek/erk + pi3k/akt
- ode model (not a fully curated biochemical map; intended for exploration + intuition)
- streamlit app with pathway graph + time courses + dose-response + sensitivity

quickstart
- create env + install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- run app: `streamlit run streamlit_app.py`

features (high level)
- pathway visualization: directed graph with node “activity” overlay from simulation state
- simulation: time courses, peak + steady-state metrics, export csv
- dose-response: sweep ligand dose and compute response curves (peak/steady)
- sensitivity: one-at-a-time local sensitivities over parameters for chosen metric

notes
- this repo is intentionally code-heavy and minimal on prose.
- parameters are chosen for qualitative behavior (transients, saturation, feedback), not absolute fit.
