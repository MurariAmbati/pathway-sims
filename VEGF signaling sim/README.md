# vegf signaling (slug: vegf)

a streamlit-based vegf-a/vegfr2 signaling simulator focused on two phenotype readouts:

- angiogenesis (sprouting / growth proxy)
- vascular permeability (barrier leak proxy)

this app is meant for interactive hypothesis exploration. it is a compact, phenomenological ode model (not a full mechanistic reconstruction).

## quickstart

1. create and activate a python environment (venv or conda)
2. install dependencies:

   pip install -r requirements.txt

3. run the app:

   streamlit run app.py

## model overview

the model tracks these normalized state variables over time:

- vegf_free: extracellular vegf-a input pool
- vegfr2_free: available receptor at the membrane
- vegfr1_free: vegfr1 decoy receptor pool (binds vegf, minimal signaling here)
- nrp1_free: neuropilin-1 co-receptor pool (facilitates vegfr2 signaling)
- vegf_bound: ligand-receptor complex
- vegf_bound_r1: vegf bound to vegfr1 (decoy complex)
- pvegfr2: activated receptor (proxy for phosphorylation)
- vegfr2_internal: internalized receptor pool
- perk: mapk/erk module activity
- pakt: pi3k/akt module activity
- psrc: src module activity (permeability-associated)
- no: nitric oxide proxy (akt -> enos -> no)
- permeability: vascular permeability proxy
- angiogenesis: angiogenesis proxy

core interactions:

- vegf binds vegfr2, forms complex, drives receptor activation
- vegf also binds vegfr1 (decoy), reducing vegf availability to vegfr2
- nrp1 increases effective vegfr2 activation (co-receptor gain)
- activated receptor drives erk, akt, and src modules
- akt produces no
- permeability is driven by src and no, and relaxes back down
- angiogenesis is driven by erk and akt, and relaxes back down
- erk provides negative feedback on receptor activation (attenuation)
- akt increases effective internalization rate (feedback)

all variables are bounded to roughly [0, 1] by construction (saturating activation terms), while vegf and vegfr2 pools can vary with user input.

## visualization

the app includes:

- interactive time courses (receptor, downstream, outputs)
- a small signaling network view (node sizes/colors reflect selected-time values)
- a 2d sweep (vegf0 vs vegfr20) with heatmaps for final permeability and angiogenesis
- csv export of the full simulated trajectory

## files

- app.py: streamlit ui and analysis workflow
- src/vegf_model.py: ode model + simulation routine
- src/vegf_viz.py: plotly/network visual helpers
- requirements.txt: python dependencies

## notes

- this is not clinical advice and is not validated for decision making.
- for biological extensions (vegfr1 decoy, neuropilin, vegf isoforms, dll4-notch tip/stalk selection, junctional dynamics), the current structure is intended to be a starting point.
