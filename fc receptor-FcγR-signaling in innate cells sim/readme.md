# fc receptor (fcγr) signaling in innate cells

comprehensive computational simulator for antibody-dependent phagocytosis and cytotoxicity mediated by fc gamma receptors in innate immune cells.

## overview

this project provides an in-depth, mechanistic simulation platform for studying fc receptor (fcγr) signaling pathways in innate immune cells, with particular emphasis on:

- **antibody-dependent cellular phagocytosis (adcp)**: mechanistic modeling of opsonization, target recognition, engulfment, and digestion
- **antibody-dependent cellular cytotoxicity (adcc)**: comprehensive simulation of target killing kinetics by nk cells, macrophages, and neutrophils
- **fcγr signaling cascade**: detailed ode-based models of itam phosphorylation, syk activation, and downstream pathways (pi3k/akt, mapk, calcium signaling)
- **receptor clustering dynamics**: simulation of fcγr crosslinking and cluster formation
- **cell type specificity**: models tailored for different innate immune cell types with distinct fcγr expression profiles

## biological background

### fc gamma receptors (fcγr)

fcγr are cell surface receptors that bind the fc region of immunoglobulin g (igg) antibodies. they are critical mediators of antibody-dependent immune responses and are expressed on various innate immune cells including:

- **macrophages**: express fcγri, fcγriia, fcγriib, fcγriiia
- **neutrophils**: express fcγriia, fcγriib, fcγriiib
- **monocytes**: express fcγri, fcγriia, fcγriib, fcγriiia
- **natural killer (nk) cells**: express fcγriiia (cd16a)
- **dendritic cells**: express fcγri, fcγriia, fcγriib

### fcγr classification

#### activating receptors
- **fcγri (cd64)**: high-affinity receptor (kd ~10⁻⁹ m), binds monomeric igg
- **fcγriia (cd32a)**: low-affinity receptor (kd ~10⁻⁶ m), contains itam motifs
- **fcγriiia (cd16a)**: low-affinity receptor, primarily on nk cells and macrophages
- **fcγriiib (cd16b)**: gpi-anchored receptor on neutrophils

#### inhibitory receptor
- **fcγriib (cd32b)**: contains itim motif, recruits ship phosphatase, dampens immune responses

### signaling mechanisms

#### itam signaling (activating receptors)
1. **receptor crosslinking**: immune complexes induce fcγr clustering
2. **itam phosphorylation**: src family kinases (lyn, fyn) phosphorylate tyrosines in itam motifs
3. **syk recruitment**: tandem sh2 domains of syk bind phospho-itam
4. **syk activation**: conformational change and auto-phosphorylation activate syk kinase
5. **downstream signaling**:
   - **pi3k pathway**: pip3 production → akt activation → cell survival, metabolism
   - **plcγ pathway**: ip3 generation → ca²⁺ release → degranulation, cytokine production
   - **mapk cascade**: mek/erk activation → transcription factor activation
   - **transcription factors**: nf-κb, nfat, ap-1 → pro-inflammatory gene expression

#### itim signaling (inhibitory receptor)
1. **fcγriib engagement**: co-ligation with activating receptors
2. **itim phosphorylation**: itim motif phosphorylation
3. **ship recruitment**: ship-1/2 phosphatases hydrolyze pip3
4. **signal dampening**: reduced pi3k/akt signaling, decreased cellular activation

### antibody-dependent cellular phagocytosis (adcp)

adcp is the process by which phagocytes (macrophages, neutrophils, monocytes) engulf and destroy antibody-opsonized targets:

#### mechanism
1. **opsonization**: igg antibodies bind to antigens on target cell surface
2. **recognition**: fcγr on phagocytes recognize fc regions of bound antibodies
3. **adhesion**: multiple fcγr-igg interactions form stable effector-target conjugates
4. **signaling**: itam-mediated signaling activates phagocytic machinery
5. **actin reorganization**: formation of pseudopodia around target
6. **phagosome formation**: target enclosed in membrane-bound phagosome
7. **phagosome maturation**: fusion with lysosomes
8. **target degradation**: proteolytic enzymes and reactive oxygen species destroy target

#### key factors affecting adcp
- **antibody concentration**: higher [igg] → increased opsonization
- **antibody isotype/subclass**: igg1, igg3 > igg2, igg4 for human fcγr
- **antigen density**: more epitopes → better opsonization
- **fcγr expression**: cell-type specific expression patterns
- **fcγr affinity**: fcγri (high) vs fcγriia/iiia (low)
- **ratio of activating:inhibitory receptors**: determines net signal

### antibody-dependent cellular cytotoxicity (adcc)

adcc is the direct killing of antibody-opsonized target cells, primarily mediated by nk cells but also by macrophages and neutrophils:

#### mechanism
1. **target recognition**: fcγriiia on nk cells binds igg-opsonized targets
2. **immune synapse formation**: stable effector-target conjugate
3. **granule polarization**: cytotoxic granules move toward synapse
4. **degranulation**: perforin and granzymes released into synapse
5. **perforin pore formation**: perforin polymerizes in target membrane
6. **granzyme entry**: serine proteases enter target cell through perforin pores
7. **apoptosis induction**: granzyme b activates caspases → programmed cell death
8. **target death**: apoptotic/necrotic cell death
9. **serial killing**: nk cell detaches and engages new target

#### cytotoxic mechanisms
- **perforin/granzyme pathway**: main mechanism for nk cells
- **fas/fasl pathway**: death receptor engagement
- **tnf-α**: pro-apoptotic cytokine
- **reactive oxygen/nitrogen species**: oxidative damage (neutrophils, macrophages)

#### factors affecting adcc
- **antibody isotype**: igg1, igg3 best for adcc
- **fcγr polymorphisms**: fcγriiia-158v/f affects affinity for igg
- **effector cell type**: nk cells > macrophages > neutrophils
- **effector:target ratio**: higher e:t → greater cytotoxicity
- **target cell sensitivity**: resistance mechanisms vary

### clinical relevance

#### therapeutic antibodies
- **rituximab (anti-cd20)**: lymphoma treatment, relies heavily on adcc/adcp
- **trastuzumab (anti-her2)**: breast cancer, adcc contributes to efficacy
- **cetuximab (anti-egfr)**: colorectal cancer, adcc-mediated
- **alemtuzumab (anti-cd52)**: chronic lymphocytic leukemia

#### fc engineering
- **afucosylation**: enhanced fcγriiia binding → improved adcc
- **amino acid substitutions**: s239d/i332e enhance fcγr binding
- **glycosylation optimization**: alters fc structure and fcγr affinity

#### autoimmune diseases
- **immune thrombocytopenia (itp)**: anti-platelet antibodies → adcp
- **autoimmune hemolytic anemia**: anti-erythrocyte antibodies → adcp
- **pemphigus**: anti-desmoglein antibodies

## features

### computational models

#### 1. fcγr signaling cascade (ode-based)
- receptor-immune complex binding kinetics
- itam phosphorylation dynamics
- syk kinase activation
- pi3k/akt pathway
- mapk cascade (mek/erk)
- plcγ/calcium signaling
- transcription factor activation (nfat, nf-κb)
- cytokine production (tnf-α, il-6, il-1β)
- inhibitory signaling (fcγriib/ship)

#### 2. antibody-dependent cellular phagocytosis
- dose-response curves for antibody concentration
- cell-type specific phagocytic efficiency
- multi-stage kinetics: binding → engulfment → digestion
- effector:target ratio effects
- antigen density dependence
- fcγr density effects

#### 3. antibody-dependent cellular cytotoxicity
- dose-response curves for antibody-mediated killing
- effector cell type comparison (nk, macrophage, neutrophil)
- cytotoxicity kinetics: conjugate formation → killing → detachment
- specific lysis calculations
- effector exhaustion modeling
- serial killing capacity

#### 4. fcγr clustering dynamics
- receptor crosslinking by immune complexes
- cluster size distribution evolution
- cooperative signaling amplification
- spatial organization effects

### visualization tools

- **interactive plotly dashboards**: real-time exploration of simulation results
- **signaling cascade visualizations**: multi-panel time-course plots
- **network diagrams**: pathway connectivity with activating/inhibitory edges
- **dose-response curves**: ec50 determination and analysis
- **kinetic plots**: time-resolved cellular processes
- **heatmaps**: parameter sensitivity analysis
- **comparative plots**: cell type and condition comparisons

### streamlit application

comprehensive web interface with:
- **overview**: biological background and pathway architecture
- **signaling cascade simulator**: customize parameters, visualize dynamics
- **adcp simulator**: optimize antibody dosing, predict phagocytic efficiency
- **adcc simulator**: model cytotoxicity kinetics, compare effector cells
- **clustering analysis**: explore receptor organization effects
- **parameter sensitivity**: identify critical rate constants
- **cell type comparison**: contrast fcγr expression and function
- **documentation**: equations, parameters, references

## installation

### requirements

```bash
python >= 3.8
streamlit >= 1.29.0
numpy >= 1.24.3
pandas >= 2.0.3
matplotlib >= 3.7.2
seaborn >= 0.12.2
plotly >= 5.17.0
scipy >= 1.11.2
networkx >= 3.1
```

### setup

```bash
# clone or navigate to project directory
cd "fc receptor-FcγR-signaling in innate cells sim"

# create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on windows: venv\\Scripts\\activate

# install dependencies
pip install -r requirements.txt
```

## usage

### run streamlit application

```bash
streamlit run app.py
```

the application will open in your default web browser at `http://localhost:8501`

### python api usage

#### example 1: simulate fcγr signaling cascade

```python
import numpy as np
from fcgr_models import FcgRSignalingModel, FcgRParameters

# create model with default parameters
params = FcgRParameters()
model = FcgRSignalingModel(params)

# simulation parameters
ic_concentration = 100e-9  # 100 nm immune complexes
time_points = np.linspace(0, 300, 1000)  # 5 minutes

# run simulation
results = model.signaling_cascade(
    ic_concentration,
    time_points,
    include_inhibitory=true
)

# access results
syk_activity = results['active_syk']
calcium_levels = results['calcium']
tnf_production = results['tnf_alpha']
```

#### example 2: model adcp efficiency

```python
from fcgr_models import AntibodyDependentPhagocytosis

model = AntibodyDependentPhagocytosis()

# dose-response curve
antibody_conc = np.logspace(-1, 2, 50)  # 0.1 to 100 μg/ml
efficiency = model.phagocytosis_efficiency(
    antibody_concentration=antibody_conc,
    antigen_density=10000,
    fcgr_density=50000,
    cell_type='macrophage'
)

# phagocytosis kinetics
time_points = np.linspace(0, 60, 500)  # 60 minutes
kinetics = model.phagocytosis_kinetics(
    opsonization_level=0.8,
    time_points=time_points,
    effector_target_ratio=10.0
)

phagocytosed_fraction = kinetics['total_phagocytosed']
```

#### example 3: simulate adcc

```python
from fcgr_models import AntibodyDependentCytotoxicity

model = AntibodyDependentCytotoxicity()

# dose-response
antibody_conc = np.logspace(-1, 2, 50)
efficiency = model.adcc_efficiency(
    antibody_concentration=antibody_conc,
    target_density=10000,
    effector_type='nk_cell'
)

# cytotoxicity kinetics
time_points = np.linspace(0, 6, 500)  # 6 hours
kinetics = model.cytotoxicity_kinetics(
    opsonization_level=0.9,
    time_points=time_points,
    effector_target_ratio=10.0,
    effector_type='nk_cell'
)

specific_lysis = kinetics['specific_lysis']
```

#### example 4: fcγr clustering

```python
from fcgr_models import FcgRCrosslinking

model = FcgRCrosslinking()

time_points = np.linspace(0, 120, 500)
results = model.clustering_dynamics(
    ic_density=5.0,
    time_points=time_points
)

average_cluster_size = results['average_cluster_size']
large_clusters = results['large_clusters']
```

#### example 5: visualization

```python
from visualization import SignalingVisualizer

visualizer = SignalingVisualizer()

# plot signaling cascade
fig = visualizer.plot_signaling_cascade(
    data=results,
    time_points=time_points,
    title="fcγr signaling dynamics"
)
fig.show()

# plot dose-response
fig = visualizer.plot_dose_response(
    antibody_conc=antibody_conc,
    efficiency=efficiency,
    ylabel="phagocytosis efficiency"
)
fig.show()

# plot pathway network
fig = visualizer.plot_pathway_network(include_inhibitory=true)
fig.show()
```

## project structure

```
fc receptor-fcγr-signaling in innate cells sim/
├── app.py                      # main streamlit application
├── fcgr_models.py             # computational models (odes, kinetics)
├── visualization.py           # plotting and visualization utilities
├── requirements.txt           # python dependencies
├── readme.md                  # this file
├── .streamlit/
│   └── config.toml           # streamlit configuration
└── [additional modules as needed]
```

## model specifications

### ordinary differential equations (odes)

the signaling cascade is modeled using a system of coupled odes:

#### receptor binding
```
d[fcγr-ic]/dt = k_on × [ic] × [fcγr_free] - k_off × [fcγr-ic]
```

#### itam phosphorylation
```
d[p-itam]/dt = k_phos × [fcγr-ic] × [itam] - k_dephos × [p-itam]
```

#### syk activation
```
d[syk*]/dt = k_syk × [p-itam] × [syk] - k_-syk × [syk*]
```

#### pi3k/akt pathway
```
d[pi3k*]/dt = k_pi3k × [syk*] × [pi3k] - k_-pi3k × [pi3k*]
d[akt*]/dt = k_akt × [pi3k*] × [akt] - k_-akt × [akt*]
```

#### mapk cascade
```
d[mek*]/dt = k_mek × [syk*] × [mek] - k_-mek × [mek*]
d[erk*]/dt = k_erk × [mek*] × [erk] - k_-erk × [erk*]
```

#### calcium signaling
```
d[plcγ*]/dt = k_plcg × [syk*] × [plcγ] - k_-plcg × [plcγ*]
d[ip3]/dt = k_ip3 × [plcγ*] - k_deg × [ip3]
d[ca²⁺]/dt = k_release × [ip3] × ([ca²⁺]_er - [ca²⁺]) - k_uptake × ([ca²⁺] - [ca²⁺]_basal)
```

#### transcription factors
```
d[nfat*]/dt = k_nfat × f(ca²⁺) × [nfat] - k_-nfat × [nfat*]
d[nf-κb*]/dt = k_nfkb × f(erk*) × [nf-κb] - k_-nfkb × [nf-κb*]
```

#### cytokines
```
d[tnf-α]/dt = k_tnf × [nf-κb*] × (1 + [nfat*])
d[il-6]/dt = k_il6 × [nf-κb*] × (1 + [akt*])
d[il-1β]/dt = k_il1b × [nf-κb*]
```

### parameter values

#### kinetic rate constants
- **itam phosphorylation**: k_phos = 0.1 s⁻¹, k_dephos = 0.05 s⁻¹
- **syk activation**: k_syk = 0.5 s⁻¹, k_-syk = 0.1 s⁻¹
- **pi3k activation**: k_pi3k = 0.3 s⁻¹, k_-pi3k = 0.15 s⁻¹
- **akt activation**: k_akt = 0.4 s⁻¹, k_-akt = 0.2 s⁻¹
- **mek activation**: k_mek = 0.35 s⁻¹, k_-mek = 0.18 s⁻¹
- **erk activation**: k_erk = 0.45 s⁻¹, k_-erk = 0.22 s⁻¹

#### receptor parameters
- **fcγr density**: 50,000 receptors/cell (macrophages)
- **kd (fcγr-igg)**: 1 × 10⁻⁷ m (low-affinity receptors)
- **kd (fcγri-igg)**: 1 × 10⁻⁹ m (high-affinity)

#### calcium parameters
- **basal ca²⁺**: 0.1 μm
- **er ca²⁺**: 10 μm
- **release rate**: k_release = 2.0 s⁻¹
- **uptake rate**: k_uptake = 1.5 s⁻¹

### phagocytosis model

efficiency calculated using michaelis-menten-like kinetics:

```
efficiency = (opsonization × s_net) / (ec50 + opsonization × s_net)
```

where:
- **opsonization**: antibody coating level (0-1)
- **s_net**: net activating signal (activating - inhibitory fcγr)
- **ec50**: half-maximal concentration (typically 0.3-0.5)

### cytotoxicity model

specific lysis calculated as:

```
specific lysis (%) = 100 × [dead targets] / [total targets]
```

kinetics follow multi-step process:
1. conjugate formation: rate ∝ opsonization × [effector] × [target]
2. killing: rate ∝ [conjugates]
3. detachment: rate ∝ [conjugates]

## parameter sensitivity

key parameters affecting outcomes:

### signaling cascade
- **itam phosphorylation rate**: directly controls signal initiation
- **syk activation rate**: amplifies downstream signaling
- **fcγriib ratio**: balances activating vs inhibitory signals
- **calcium release rate**: affects degranulation and cytokine production

### adcp
- **antibody concentration**: primary driver of opsonization
- **antigen density**: more epitopes → better opsonization
- **fcγr density**: higher expression → stronger phagocytic signal
- **effector:target ratio**: more effectors → faster phagocytosis

### adcc
- **antibody concentration**: critical for target opsonization
- **effector type**: nk cells > macrophages > neutrophils
- **effector:target ratio**: higher ratio → greater cytotoxicity
- **target sensitivity**: intrinsic susceptibility varies

## validation and benchmarking

models are parameterized based on:

### experimental data sources
- **binding kinetics**: surface plasmon resonance (spr) measurements
- **calcium imaging**: confocal microscopy time-series
- **flow cytometry**: receptor expression levels, phosphorylation states
- **live-cell imaging**: phagocytosis and cytotoxicity kinetics
- **cytokine elisa**: tnf-α, il-6, il-1β production

### literature values
- nimmerjahn & ravetch (2008): fcγr biology and signaling
- wang et al. (2015): antibody engineering effects
- overdijk et al. (2012): adcp mechanisms and kinetics
- bournazos & ravetch (2017): adcc in therapeutic contexts

## applications

### research applications
- **mechanism exploration**: dissect signaling pathway components
- **hypothesis testing**: predict effects of pathway perturbations
- **drug target identification**: identify rate-limiting steps
- **biomarker discovery**: correlate molecular states with functional outcomes

### therapeutic development
- **antibody optimization**: predict effects of fc engineering
- **dosing strategies**: optimize antibody concentrations
- **combination therapies**: model synergistic effects
- **resistance mechanisms**: understand escape pathways

### educational uses
- **immunology teaching**: visualize complex signaling cascades
- **systems biology training**: demonstrate ode-based modeling
- **data interpretation**: connect molecular events to cellular outcomes

## limitations

### model assumptions
- **well-mixed system**: ignores spatial heterogeneity
- **deterministic odes**: neglects stochastic fluctuations
- **simplified geometry**: no membrane organization details
- **population average**: cell-to-cell variability not captured
- **limited crosstalk**: focused on fcγr-specific pathways

### biological complexity not included
- **other immune receptors**: tlrs, complement receptors, etc.
- **metabolic constraints**: energy requirements
- **cell cycle effects**: proliferation and differentiation
- **tissue microenvironment**: cytokines, hypoxia, ph
- **adaptive immune interactions**: t cell help, memory responses

## future enhancements

### planned features
- **stochastic modeling**: gillespie algorithm for molecular noise
- **spatial models**: pde-based simulations with diffusion
- **multi-scale integration**: link molecular → cellular → tissue levels
- **machine learning**: parameter fitting from experimental data
- **uncertainty quantification**: bayesian parameter estimation

### additional cell types
- **eosinophils**: fcεri and fcγr cooperation
- **basophils**: mediator release
- **mast cells**: allergic responses

### expanded pathways
- **complement system**: c3b opsonization synergy
- **toll-like receptors**: pattern recognition co-stimulation
- **cytokine networks**: il-12, ifn-γ feedback loops

## contributing

contributions are welcome! areas for improvement:

- **parameter refinement**: incorporate latest literature values
- **model validation**: compare with experimental datasets
- **new features**: add cell types, pathways, or analysis tools
- **documentation**: improve explanations and examples
- **testing**: unit tests for models and visualizations

## license

this project is provided for educational and research purposes. please cite appropriately if used in publications.

## citations

if you use this simulator in your research, please cite:

### key references for model development

1. nimmerjahn, f. & ravetch, j.v. (2008). fcγ receptors as regulators of immune responses. *nature reviews immunology* 8:34-47.

2. guilliams, m. et al. (2014). the function of fcγ receptors in dendritic cells and macrophages. *nature reviews immunology* 14:94-108.

3. overdijk, m.b. et al. (2012). antibody-mediated phagocytosis contributes to the anti-tumor activity of the therapeutic antibody daratumumab in lymphoma and multiple myeloma. *mabs* 7:311-321.

4. bournazos, s. & ravetch, j.v. (2017). fcγ receptor function and the design of vaccination strategies. *immunity* 47:224-233.

5. wang, x. et al. (2015). igg fc engineering to modulate antibody effector functions. *protein & cell* 9:63-73.

## contact

for questions, bug reports, or suggestions, please open an issue in the project repository.

## acknowledgments

this simulator integrates knowledge from decades of immunology research on fc receptors, antibody effector functions, and innate immunity. we acknowledge the scientific community whose experimental work provided the foundation for these computational models.

---

**slug**: fc_receptor  
**version**: 1.0.0  
**last updated**: 2024
