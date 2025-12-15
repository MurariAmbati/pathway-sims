# calcium (ca²⁺) signaling

streamlit app for simulating compact ca²⁺ second‑messenger signaling (ca²⁺ cytosol + er, ip3, cam, pkc) with plots and metrics.

## run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## outputs

- pathway diagram (highlight + inspector)
- time series: ca²⁺ (cyt/er), ip3, cam*, pkc*, fluxes
- phase plots + fft spectrum
- metrics table per signal: baseline, peak, t_peak, Δpeak, auc, rise 10–90%, decay τ, peak count, dominant fft
