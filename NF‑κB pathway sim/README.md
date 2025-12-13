# nfkb

nf‑κb pathway simulation (ode + stochastic) with oscillatory negative feedback.

- slug: nfkb
- role: inflammation, immune signaling, stress responses
- style: ode with oscillatory dynamics; optional noise variant

## quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## streamlit ui

single page, parameter-driven.

- model: nf‑κb nuclear import/export + iκb transcription/translation/degradation (negative feedback)
- input: ikk activity (constant or pulse)
- modes:
  - deterministic (ode): `solve_ivp` (rk45/radau/bdf)
  - stochastic: euler‑maruyama additive noise

## visualizations

- time series: nn, im, i, and ikk(t)
- phase portrait: nn vs i
- spectrum: fft power spectrum of nn (dominant period estimate)
- data: table export/view

## notes

this is a compact, qualitative oscillator intended for exploration and visualization (not a curated, parameter-fitted biological reference model).
