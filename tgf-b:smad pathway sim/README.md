# tgfb_smad

interactive ode model for tgf-β → smad2/3/4 with smad7 feedback and mapk/pi3k crosstalk.

## run

macos note: this folder name contains `:` so python `venv` will not create an env inside it. create the venv next to the project folder.

```bash
cd "/Users/murari/biosim"
/opt/homebrew/bin/python3 -m venv tgfb_smad_venv
source tgfb_smad_venv/bin/activate
pip install -r "/Users/murari/biosim/tgf-b:smad pathway sim/requirements.txt"

cd "/Users/murari/biosim/tgf-b:smad pathway sim"
streamlit run app.py
```

## outputs

- time-course plots (receptor, smad phosphorylation, nuclear complex, programs)
- dose–response curves
- mapk×pi3k crosstalk heatmap
- csv download buttons for each analysis
