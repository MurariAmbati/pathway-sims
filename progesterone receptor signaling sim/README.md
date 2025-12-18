# Progesterone Receptor Signaling

Computational simulation and analysis of progesterone receptor-mediated cellular signaling.

## Overview

This application models the molecular dynamics of progesterone receptor (PR) signaling across different physiological contexts, including pregnancy, uterine function, and breast development. The simulation captures receptor binding kinetics, nuclear translocation, DNA binding, gene transcription, and downstream cellular responses.

## Features

- **Receptor Dynamics Modeling**: ODE-based simulation of PR-A and PR-B isoforms
- **Tissue-Specific Models**: Pre-configured parameters for uterine, breast, and pregnancy contexts
- **Dose-Response Analysis**: EC50 determination and Hill coefficient calculation
- **Gene Expression Profiling**: Quantitative analysis of progesterone-regulated genes
- **Pathway Visualization**: Interactive network diagrams and signaling cascade plots
- **Cellular Response Metrics**: Decidualization, implantation receptivity, and proliferation indices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Model Architecture

### Receptor States
- Cytoplasmic unbound receptors (PR-A, PR-B)
- Ligand-bound complexes (PR:P4)
- Homodimers and heterodimers
- Nuclear receptors
- DNA-bound complexes with coactivators

### Key Parameters
- Binding kinetics: k_on, k_off, K_d
- Dimerization rates
- Nuclear import/export
- Transcription and translation rates
- mRNA and protein degradation

### Tissue Models
- **Uterine**: High PR-B expression, enhanced transcriptional activity
- **Breast**: Balanced PR-A/PR-B isoforms
- **Pregnancy**: Trimester-specific receptor expression levels

## Scientific Basis

The model incorporates:
- Ligand binding thermodynamics
- Receptor dimerization equilibria
- Nuclear-cytoplasmic shuttling
- Progesterone response element (PRE) binding
- Coactivator recruitment dynamics
- MAPK and PI3K crosstalk pathways

## Structure

```
├── app.py                          # Streamlit interface
├── models/
│   ├── receptor_dynamics.py       # Core ODE model
│   └── downstream_signaling.py    # Gene regulation and pathways
├── visualizations/
│   └── plots.py                    # Plotly-based visualizations
└── requirements.txt
```

## License

MIT
