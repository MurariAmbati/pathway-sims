# type i interferon (ifn-α/β) signaling pathway simulator

interactive simulation and visualization platform for exploring type i interferon signaling dynamics in antiviral innate immunity.

## overview

this project provides a comprehensive computational model of the ifn-α/β signaling pathway, featuring:

- detailed ode-based mechanistic model
- interactive streamlit web interface
- multiple visualization modes
- dose-response analysis
- parameter sensitivity exploration
- real-time dynamic simulations

## biological background

type i interferons (ifn-α and ifn-β) are critical cytokines in the innate immune response to viral infections. the signaling cascade involves:

1. **receptor binding**: ifn-α/β binds to ifnar1/ifnar2 heterodimeric receptor complex
2. **jak activation**: receptor-associated jak1 and tyk2 kinases become activated
3. **stat phosphorylation**: jak kinases phosphorylate stat1 and stat2 transcription factors
4. **isgf3 formation**: phosphorylated stat1-stat2 heterodimer associates with irf9
5. **nuclear translocation**: isgf3 complex enters nucleus
6. **isg expression**: isgf3 binds isre elements, inducing interferon-stimulated genes (isgs)
7. **antiviral state**: isg proteins establish cellular antiviral defenses

### feedback regulation

- **negative feedback**: socs proteins inhibit jak activity, limiting signal duration
- **positive feedback**: isgs include ifn-β, amplifying the response

## installation

### requirements

- python 3.8+
- dependencies listed in `requirements.txt`

### setup

```bash
# clone or download the repository
cd "type I interferon-IFN‑α:β-signaling sim"

# create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## usage

### running the streamlit app

```bash
streamlit run app.py
```

the app will open in your default web browser at `http://localhost:8501`

### basic workflow

1. **configure parameters** in the sidebar:
   - simulation duration
   - initial ifn concentration
   - receptor levels
   - advanced kinetic parameters (optional)

2. **run simulation** by clicking the "run simulation" button

3. **explore results** across multiple tabs:
   - **overview**: key metrics and main time course
   - **detailed dynamics**: signaling cascade progression
   - **phase analysis**: phase portraits of component relationships
   - **network view**: pathway architecture diagram
   - **dose-response**: ifn concentration effects
   - **analysis**: comprehensive metrics and data export

### using the python modules directly

```python
from pathway_model import IFNSignalingModel, PathwayParameters

# create model with default parameters
model = IFNSignalingModel()

# or customize parameters
params = PathwayParameters()
params.ifn_initial = 200.0
params.k_jak_phosph = 0.7
model = IFNSignalingModel(params)

# run simulation
t, solution = model.simulate((0, 200), n_points=1000)

# analyze results
from pathway_model import analyze_pathway_dynamics
analysis = analyze_pathway_dynamics(model, solution, t)
print(analysis)
```

### visualization examples

```python
from visualizations import plot_pathway_timecourse, plot_signaling_cascade

# create time course plot
fig = plot_pathway_timecourse(t, solution, model.state_names)
fig.show()

# create cascade visualization
fig = plot_signaling_cascade(t, solution, model.state_names)
fig.show()
```

## model components

### state variables (18 total)

| variable | description |
|----------|-------------|
| ifn | free interferon concentration |
| ifnar | free receptor concentration |
| ifn_ifnar | bound receptor complex |
| ifnar_intern | internalized receptors |
| jak_inactive | inactive jak kinases |
| jak_active | active jak kinases |
| stat1, stat2 | unphosphorylated stats |
| pstat1, pstat2 | phosphorylated stats |
| stat1_stat2 | stat heterodimer |
| irf9 | irf9 protein |
| isgf3_cyto | isgf3 in cytoplasm |
| isgf3_nuc | isgf3 in nucleus |
| isg_mrna | interferon-stimulated gene mrna |
| isg_protein | isg protein products |
| socs | suppressor of cytokine signaling |
| antiviral_state | cumulative antiviral capacity |

### key parameters

#### receptor dynamics
- `k_ifn_bind`: ifn-receptor binding rate
- `k_ifn_unbind`: unbinding rate
- `k_receptor_intern`: receptor internalization rate

#### signaling activation
- `k_jak_phosph`: jak phosphorylation rate
- `k_stat_phosph`: stat phosphorylation rate
- `k_stat_dimer`: stat dimerization rate

#### gene expression
- `k_isg_transcr`: isg transcription rate
- `k_protein_synth`: protein synthesis rate

#### feedback
- `k_socs_synth`: socs synthesis rate
- `k_socs_inhib`: socs inhibition strength
- `k_ifn_feedback`: positive feedback strength

## features

### interactive controls
- real-time parameter adjustment
- multiple initial condition scenarios
- advanced parameter fine-tuning

### visualizations
- **time course plots**: track component dynamics over time
- **signaling cascade**: view progression through pathway stages
- **phase portraits**: explore relationships between components
- **network diagrams**: understand pathway architecture
- **dose-response curves**: analyze concentration effects
- **comparative analysis**: compare different conditions

### analysis tools
- peak detection and timing
- response time calculation
- steady-state analysis
- data export (csv format)

## biological insights

### clinical relevance

**therapeutic applications:**
- ifn-α therapy for hepatitis c
- cancer immunotherapy
- multiple sclerosis treatment

**disease mechanisms:**
- viral immune evasion strategies
- autoimmune disease signatures
- interferon-mediated pathology

### research applications

- hypothesis testing for pathway perturbations
- drug target identification
- biomarker discovery
- systems biology education

## file structure

```
type I interferon-IFN‑α:β-signaling sim/
├── app.py                    # main streamlit application
├── pathway_model.py          # ode model implementation
├── visualizations.py         # plotting functions
├── requirements.txt          # python dependencies
└── readme.md                # this file
```

## technical details

### numerical methods
- ode integration: scipy's `odeint` (lsoda algorithm)
- adaptive time stepping for accuracy
- mass-action and michaelis-menten kinetics

### performance
- typical simulation: <2 seconds for 1000 time points
- dose-response analysis: ~20-30 seconds for 20 doses
- responsive interactive interface

## customization

### adding new parameters

edit `PathwayParameters` class in `pathway_model.py`:

```python
@dataclass
class PathwayParameters:
    # add new parameter
    k_new_process: float = 0.5
```

### modifying the model

update the `derivatives` method in `IFNSignalingModel`:

```python
def derivatives(self, state, t):
    # add new reactions or modify existing ones
    d_new_component = ...
    return np.array([...])
```

### creating custom visualizations

add functions to `visualizations.py`:

```python
def plot_custom_analysis(...):
    fig = go.Figure()
    # create visualization
    return fig
```

## troubleshooting

**simulation runs slowly:**
- reduce number of time points
- decrease simulation duration
- disable advanced visualizations temporarily

**numerical instabilities:**
- adjust parameter values (avoid extreme values)
- modify time step size
- check for biologically unrealistic parameters

**visualization issues:**
- ensure plotly is properly installed
- check browser compatibility
- clear streamlit cache

## references

### key publications

1. schoggins, j.w. (2019). interferon-stimulated genes: what do they all do? *annu rev virol* 6:567-584.

2. ivashkiv, l.b. & donlin, l.t. (2014). regulation of type i interferon responses. *nat rev immunol* 14:36-49.

3. schneider, w.m., chevillotte, m.d., & rice, c.m. (2014). interferon-stimulated genes: a complex web of host defenses. *annu rev immunol* 32:513-545.

4. platanias, l.c. (2005). mechanisms of type-i- and type-ii-interferon-mediated signalling. *nat rev immunol* 5:375-386.

### pathway databases

- kegg pathway: hsa04620 (toll-like receptor signaling)
- reactome: r-hsa-909733 (interferon alpha/beta signaling)
- wikipathways: wp585 (ifn-alpha/beta signaling)

## license

this project is provided for educational and research purposes.

## contributing

suggestions for improvements:

- additional feedback mechanisms
- cell type-specific parameters
- spatial modeling components
- stochastic simulation options
- machine learning integration

## contact & support

for questions, suggestions, or issues, please refer to standard python/streamlit documentation and biological pathway resources.

## acknowledgments

developed for exploring the complex dynamics of interferon signaling in innate immunity and antiviral responses.

---

**keywords:** type i interferon, ifn-alpha, ifn-beta, jak-stat signaling, innate immunity, antiviral response, systems biology, computational modeling, pathway simulation

**slug:** ifn_type1
