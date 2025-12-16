# PDGF signaling sim (fibroblast)

Streamlit app that simulates a mechanistic (ODE) PDGF→PDGFR signaling model and visualizes downstream activation and a proliferation proxy.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## What’s inside

- `app.py`: Streamlit UI + visualization
- `src/pdgf_sim/model.py`: ODE model + parameter presets
- `src/pdgf_sim/sim.py`: simulation helpers, sweeps, exports

## Deploy

Most Streamlit hosts will detect `requirements.txt`. If you need Docker, see `Dockerfile`.
