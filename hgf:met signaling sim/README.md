# hgf/met signaling simulator

interactive simulation of hepatocyte growth factor and met receptor signaling pathway modeling cell motility, invasion, and liver regeneration.

## overview

simulates 22 molecular species across multiple pathways:
- pi3k-akt (survival)
- ras-mapk (proliferation)
- stat3 (transcription)
- rac1/cdc42 (motility)

cellular outputs: invasion, motility, proliferation, survival

## installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

requirements: python 3.8+

## usage

1. set simulation parameters (duration, time points)
2. choose hgf stimulation protocol (constant, pulse, ramp, oscillatory)
3. adjust pathway rates (optional)
4. explore visualizations across tabs

## features

- ode-based mathematical modeling
- interactive network graphs
- time series and heatmaps
- phase space analysis
- dose-response curves
- parameter comparisons
- csv export

## model

### reactions
```
hgf + met ⇌ hgf:met → pmet
pmet → ppi3k → pakt → survival
pmet → pras → perk → proliferation
pmet → pstat3 → transcription
pmet → prac1/pcdc42 → motility/invasion
```

### feedback
- pakt inhibits met expression
- perk attenuates ras activation

## deployment

streamlit community cloud recommended:
1. push to github
2. connect at streamlit.io/cloud
3. deploy

alternatives: heroku, aws ec2, docker

## technical

- integration: lsoda adaptive ode solver
- frontend: streamlit
- visualization: plotly
- computation: numpy/scipy
- network: networkx

simulation time: <5s for 1000 points

## biological context

normal: wound healing, liver regeneration, development
pathological: cancer metastasis, met dysregulation
therapeutic: met inhibitors (crizotinib, capmatinib)

## troubleshooting

```bash
# dependencies
pip install --upgrade -r requirements.txt

# port conflict
streamlit run app.py --server.port 8502
```

## license

educational and research use

last updated: december 2025
