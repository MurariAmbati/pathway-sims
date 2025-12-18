# eph/ephrin signaling simulation

comprehensive computational model of eph receptor and ephrin ligand signaling dynamics in neural development, including molecular signaling cascades, tissue boundary formation, and axon guidance mechanisms.

## overview

eph receptors and ephrin ligands constitute the largest family of receptor tyrosine kinases and mediate critical developmental processes through bidirectional signaling. this simulation explores:

- **molecular signaling**: forward and reverse signaling pathways, receptor clustering, and downstream effector activation (rac/rho gtpases)
- **tissue boundaries**: cell sorting and boundary formation through differential eph/ephrin expression and adhesion dynamics
- **axon guidance**: growth cone navigation through ephrin gradients with chemotactic responses
- **topographic mapping**: retinotectal map formation via complementary gradient systems

## biological background

eph/ephrin signaling plays essential roles in:

- axon guidance and neural circuit formation
- tissue boundary establishment during segmentation
- cell migration and positioning
- vascular development
- synaptic plasticity

the system exhibits unique bidirectional signaling where both the receptor-bearing cell (forward signaling) and ligand-bearing cell (reverse signaling) receive biochemical signals upon binding.

## features

### molecular dynamics
- ode-based model of receptor-ligand binding kinetics
- receptor clustering and trans-autophosphorylation
- forward signaling pathway (eph → cell interior)
- reverse signaling pathway (ephrin → cell interior)
- downstream rac1 and rhoa activation with cross-inhibition
- endocytosis and receptor recycling

### spatial modeling
- 2d cellular potts model for cell sorting
- differential adhesion with homotypic/heterotypic interactions
- eph-ephrin mediated repulsion fields
- boundary sharpening dynamics
- quantitative metrics: segregation index, boundary sharpness, adhesion energy

### axon guidance
- growth cone chemotaxis simulation
- filopodia-based gradient sensing
- persistent random walk with chemotactic bias
- multiple gradient patterns: linear, exponential, striped, barriers
- trajectory analysis and tortuosity measurements

### topographic mapping
- retinotectal projection simulation
- opposing gradient systems (ephrina/ephrinb)
- ordered map formation
- correlation analysis and mapping precision metrics

## installation

```bash
pip install -r requirements.txt
```

## usage

run the streamlit application:

```bash
streamlit run app.py
```

navigate through the four simulation modules:

1. **molecular signaling**: explore temporal dynamics of signaling cascades
2. **tissue boundaries**: simulate cell sorting and boundary formation
3. **axon guidance**: model growth cone navigation
4. **topographic mapping**: generate retinotectal maps

## model architecture

### molecular model (`eph_ephrin_model.py`)

implements system of 10 ordinary differential equations tracking:
- free eph receptors
- free ephrin ligands  
- eph-ephrin complexes
- clustered receptor complexes
- activated eph receptors
- forward and reverse signaling strength
- active rac-gtp and rhoa-gtp
- internalized receptors

key parameters control binding kinetics, clustering rates, activation dynamics, and downstream signaling strength.

### spatial model (`spatial_model.py`)

monte carlo simulation using metropolis-like dynamics:
- cell-cell interactions via adhesion energies
- local repulsion from eph-ephrin products
- energy minimization drives cell sorting
- gaussian smoothing for molecular diffusion
- boundary detection via image gradients

### axon guidance model (`axon_guidance.py`)

agent-based growth cone simulation:
- growth cones as autonomous agents
- gradient sensing via filopodia sampling
- biased random walk based on local ephrin levels
- self-avoidance and collision detection
- persistent directional movement

## parameters

### molecular signaling
- `k_bind`, `k_unbind`: binding kinetics (determines affinity)
- `k_cluster`: receptor clustering rate (cooperative binding)
- `k_activation`: kinase activation rate
- `k_rac_act`, `k_rho_act`: downstream effector activation
- `k_rac_rho_inhib`, `k_rho_rac_inhib`: cross-talk parameters

### spatial dynamics
- `homotypic_adhesion`: same cell type attraction
- `heterotypic_adhesion`: different cell type attraction
- `repulsion_strength`: eph-ephrin repulsion magnitude
- `repulsion_range`: spatial extent of repulsion
- `diffusion_rate`: molecular field smoothing

### axon guidance
- `growth_speed`: extension velocity
- `turning_sensitivity`: response to gradients
- `persistence_length`: directional persistence
- `filopodia_length`: sensing range
- `sensing_noise`: stochasticity in detection

## biological insights

### repulsion vs attraction
the balance between rac (attractive) and rho (repulsive) activation determines cellular response. high eph-ephrin binding typically activates rho, leading to growth cone collapse and repulsion, but context-dependent signaling can produce attraction.

### boundary formation
sharp boundaries emerge from:
1. differential expression creates repulsion gradients
2. like cells preferentially adhere (homotypic > heterotypic)
3. energy minimization drives segregation
4. positive feedback sharpens interfaces

### topographic mapping
ordered neural maps form through:
1. complementary gradients in source (retina) and target (tectum)
2. axons expressing high epha avoid high ephrina regions
3. opposing gradients provide two-dimensional positional information
4. activity-dependent refinement (not modeled here) stabilizes connections

## validation

model behaviors consistent with experimental observations:

- receptor clustering required for efficient signaling activation
- bistability in rho/rac balance produces switch-like responses
- cell populations segregate within biologically relevant timescales
- axons navigate appropriately through various gradient configurations
- topographic maps show correlation coefficients > 0.8

## extensions

possible future developments:

- three-dimensional spatial models
- activity-dependent synaptic refinement
- mechanotransduction coupling
- competition between multiple axons for target space
- genetic perturbation analysis (knockout simulations)
- temporal gradient dynamics (development progression)

## references

key biological concepts implemented:

- eph/ephrin bidirectional signaling
- cellular potts models for tissue dynamics
- growth cone chemotaxis
- differential adhesion hypothesis
- chemoaffinity hypothesis (sperry, 1963)
- topographic map formation mechanisms

## technical details

**numerical methods**:
- ode integration: scipy.integrate.odeint with adaptive stepping
- spatial dynamics: metropolis monte carlo
- gradient sensing: bilinear interpolation
- smoothing: gaussian convolution (scipy.ndimage)

**performance**:
- molecular simulation: ~1000 timepoints in <1s
- spatial simulation: 100x100 grid, 200 steps in ~5s
- axon guidance: 10 growth cones, 400 steps in ~2s

**visualization**:
- matplotlib for static 2d plots
- plotly for interactive time series and phase planes
- colormaps optimized for scientific visualization

## license

this is educational/research software for exploring eph/ephrin signaling mechanisms.

## slug

eph_ephrin
