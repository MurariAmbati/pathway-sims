# nlrp3 inflammasome activation

simulation of nlrp3 inflammasome-mediated il-1β/il-18 maturation and pyroptosis.

## model

two-signal mechanism:
- signal 1: nf-κb → pro-il-1β, pro-il-18, nlrp3
- signal 2: nlrp3* + asc + pro-caspase-1 → inflammasome
- outcomes: caspase-1 → il-1β/il-18 maturation, gsdmd cleavage, pyroptosis

## run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## visualizations

- outcomes: cytokine secretion, cell viability, gsdmd dynamics
- dynamics: component time courses by stage
- heatmap: temporal patterns
- phase space: trajectory analysis
- pathway: interactive schematic

## parameters

adjustable kinetics: transcription, assembly, activation, maturation, death

signal profiles: step, pulse, ramp, oscillating
