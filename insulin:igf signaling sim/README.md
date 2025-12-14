# insulin / igf signaling sim (streamlit)

interactive insulin/igf pathway simulator (pi3k/akt + mapk) with a simplified ode model, a pathway graph, and glucose readouts.

## run (macos/homebrew)

note: this folder name contains `:` which breaks creating a venv inside the project. create the venv outside (any path without `:`).

```bash
cd "/users/murari/biosim/insulin:igf signaling sim"

/opt/homebrew/bin/python3 -m venv /users/murari/biosim/.venvs/insulin_igf
/users/murari/biosim/.venvs/insulin_igf/bin/python -m pip install -r requirements.txt

/users/murari/biosim/.venvs/insulin_igf/bin/streamlit run app.py
```

app url prints in the terminal (typically `http://127.0.0.1:8502` if 8501 is busy).

## model (toy)

- states are normalized activations in [0,1] for signaling nodes.
- includes feedback/crosstalk: s6k inhibits irs; pten reduces pip3.
- glucose is a 1-compartment variable with hepatic production and peripheral uptake modulated by akt/glut4.

not for clinical use.
