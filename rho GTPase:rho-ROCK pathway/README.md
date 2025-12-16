# rho_rock

rho gtpase / rho–rock pathway (rhoa → rock/mdia) for actin cytoskeleton dynamics, cell shape, migration, contractility.

## run

```bash
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/rho_rock
source ~/.venvs/rho_rock/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## includes

- pathway network graph (activation vs inhibition, directed)
- ode simulation with upstream drive + perturbations
- readouts: factin, pmlc, contractility proxy, migration proxy
