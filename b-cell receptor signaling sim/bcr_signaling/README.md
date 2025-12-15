# bcr signaling

slug: bcr_signaling

## what this is

a streamlit app that simulates a compact b cell receptor (bcr) signaling network using ordinary differential equations (odes).

the model captures:
- proximal bcr→syk activation
- plcγ2 → calcium → pkcβ branch
- pi3k → akt survival branch
- mapk activation marker branch
- canonical ikk/iκb/nf-κb nuclear translocation with a20-like negative feedback

all state variables are dimensionless activities (roughly 0–1). this is for mechanistic exploration and visualization, not a calibrated quantitative model.

## run

### 1) create env and install

from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) start the app

```bash
streamlit run app.py
```

## app layout

- simulate: interactive timecourses and summary metrics, csv export
- network: influence graph of the modeled interactions
- sweeps: 2d parameter sweep heatmap over a chosen output metric
- sensitivity: local log-sensitivity for a selected parameter and metric

## interventions

the sidebar supports:
- syk inhibitor
- btk inhibitor (implemented by reducing plcγ2 and pkc activation)
- pi3k inhibitor
- ikk inhibitor
- shp1 knockdown (reduces proximal phosphatase brake)
- pten knockdown (reduces pi3k brake)

## model notes

the core nf-κb negative feedback is implemented as a20 inhibiting ikk activation, and a20 production being induced by nuclear nf-κb.

## files

- app.py: streamlit ui
- src/bcr_signaling/model.py: odes + simulation + network
- src/bcr_signaling/viz.py: plotly charts + sweep/sensitivity helpers
- src/bcr_signaling/defaults.py: defaults and initial conditions
