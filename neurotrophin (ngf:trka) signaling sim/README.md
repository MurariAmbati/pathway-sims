# neurotrophin (ngf/trka) signaling simulation

**slug:** `ngf_trka`

## overview

interactive simulation and visualization of neurotrophin signaling pathways, focusing on nerve growth factor (ngf) and its high-affinity receptor tropomyosin receptor kinase a (trka). this system models key molecular events in neuronal survival and differentiation.

## biological context

neurotrophin signaling through ngf/trka is essential for:
- neuronal survival and apoptosis prevention
- axonal growth and guidance
- synaptic plasticity and maintenance
- differentiation of sympathetic and sensory neurons

## signaling pathways

### 1. ras/mapk pathway
- promotes neuronal differentiation
- activates transcription factors (elk-1, creb)
- regulates immediate early genes (c-fos, c-jun)

### 2. pi3k/akt pathway
- primary survival signaling
- inhibits pro-apoptotic factors (bad, caspase-9)
- activates mtor for protein synthesis

### 3. plcγ pathway
- calcium signaling and gene transcription
- activates protein kinase c (pkc)
- modulates synaptic function

## features

- **real-time simulation**: dynamic ode-based modeling of signaling cascades
- **interactive visualizations**: 3d pathway networks, temporal dynamics, heatmaps
- **parameter control**: adjust ligand concentrations, receptor density, kinetic rates
- **pathway analysis**: quantify signal propagation, crosstalk, and feedback loops
- **comparative studies**: multiple conditions, dose-response curves, time-course analysis

## implementation

### molecular components
- ngf ligand binding and dimerization
- trka receptor activation and phosphorylation
- adaptor proteins (shc, grb2, gab1, frs2)
- downstream kinases (raf, mek, erk, pi3k, akt)
- transcription factors and gene expression

### mathematical modeling
- ordinary differential equations (odes)
- mass action kinetics
- michaelis-menten enzyme kinetics
- receptor-ligand binding equilibria

## visualization capabilities

- network topology graphs with signal flow
- time-series plots of protein activation
- phase portraits and bifurcation diagrams
- sensitivity analysis heatmaps
- 3d spatial-temporal representations

## usage

```bash
streamlit run app.py
```

## requirements

- python 3.8+
- streamlit
- numpy, scipy
- plotly, matplotlib
- networkx
- pandas

## references

- reichardt lf (2006). neurotrophin-regulated signalling pathways. philos trans r soc lond b biol sci.
- huang ej, reichardt lf (2001). neurotrophins: roles in neuronal development and function. annu rev neurosci.
- kaplan dr, miller fd (2000). neurotrophin signal transduction in the nervous system. curr opin neurobiol.
