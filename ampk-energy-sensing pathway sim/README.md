# ampk

ampk energy-sensing pathway simulation (streamlit).

goal: capture the control logic linking cellular energy status to ampk activation, mtorc1 regulation, ulk1, and autophagy.

## quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## model

state variables (0..1 proxies):
- energy: energy charge proxy (high = atp-sufficient)
- ampk: active ampk fraction
- mtorc1: active mtorc1 fraction
- ulk1: active ulk1 fraction
- autophagy: autophagy flux proxy

inputs (0..1 unless noted):
- nutrient availability
- energy demand (0..1.5)
- optional stress pulse on demand (t0, duration, amplitude)
- optional nutrient step (t0, step-to)

logic:
- low energy activates ampk (via energy stress = 1 - energy)
- ampk inhibits mtorc1 and activates ulk1
- mtorc1 inhibits ulk1
- ulk1 drives autophagy
- ampk increases energy production capacity; mtorc1 increases energy consumption

## outputs

the app plots time series for energy/ampk/mtorc1/ulk1/autophagy plus the input trajectories, and provides csv + json export.

## scope

this is not a mechanistic, parameter-identified biochemical model; it is a stable, interpretable, ode-based pathway control model suitable for interactive exploration.
