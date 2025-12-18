# chemokine receptor (cxcr/ccr) signaling simulation

**slug:** `chemokine_receptors`

## overview

comprehensive computational model of chemokine receptor signaling pathways focusing on cxcr and ccr family receptors. simulates leukocyte trafficking, positioning, and chemotactic responses in tissue microenvironments.

## biological system

### chemokine receptors
- **cxcr family**: cxcr1-7 (primarily neutrophils, t cells, nk cells)
- **ccr family**: ccr1-10 (monocytes, dendritic cells, memory t cells)
- **g-protein coupled receptors (gpcrs)**: 7-transmembrane domain architecture
- **ligand specificity**: chemokine gradients (cxcl, ccl families)

### signaling cascades
1. **receptor activation**: chemokine binding → conformational change
2. **g-protein coupling**: gαi/o dissociation → βγ subunit release
3. **downstream pathways**:
   - pi3k/akt → actin polymerization
   - plcβ → calcium mobilization
   - mapk/erk → gene transcription
   - small gtpases (rac, cdc42) → cytoskeletal reorganization

### leukocyte trafficking
- **rolling**: selectin-mediated attachment
- **arrest**: chemokine-triggered integrin activation
- **adhesion**: firm attachment to endothelium
- **transmigration**: diapedesis through vessel wall
- **chemotaxis**: directed migration along gradients

## features

### simulation components
- multi-scale modeling (molecular → cellular → tissue)
- spatial chemokine gradient generation
- receptor binding kinetics (kon, koff, kd)
- signal transduction network dynamics
- leukocyte population behavior
- tissue compartment modeling

### visualization
- real-time streamlit dashboard
- 3d chemokine gradient heatmaps
- leukocyte trajectory tracking
- receptor occupancy dynamics
- signal pathway activation plots
- parameter sensitivity analysis

## installation

```bash
pip install -r requirements.txt
```

## usage

### run simulation
```bash
streamlit run app.py
```

### configure parameters
edit `config.yaml` for:
- receptor expression levels
- chemokine concentrations
- tissue geometry
- migration parameters
- simulation duration

## model architecture

### core modules
- `models/receptors.py`: receptor classes (cxcr, ccr)
- `models/chemokines.py`: ligand properties and gradients
- `models/signaling.py`: intracellular pathway odes
- `models/leukocytes.py`: cell agent-based model
- `models/tissue.py`: spatial environment
- `simulation/engine.py`: integration and time-stepping
- `visualization/plots.py`: analysis and figures

### mathematical framework
- **receptor dynamics**: mass action kinetics
- **signal transduction**: ordinary differential equations (odes)
- **chemotaxis**: keller-segel pde approximation
- **cell migration**: persistent random walk + biased motility
- **spatial discretization**: finite difference methods

## parameters

### key variables
- receptor density: 10^3-10^5 per cell
- chemokine kd: 1-100 nm range
- migration speed: 5-20 μm/min
- gradient sensing: ~2% concentration difference
- simulation timestep: 0.1-1.0 seconds

## scientific basis

implements published models:
- receptor-ligand binding: langmuir adsorption
- signal amplification: ultrasensitive responses
- gradient sensing: local excitation, global inhibition (legi)
- persistent migration: ornstein-uhlenbeck process

## output

- time-series data (csv/hdf5)
- trajectory files (json)
- activation heatmaps (png)
- statistical summaries (txt)
- interactive visualizations (html)

## dependencies

- numpy, scipy: numerical computation
- pandas: data handling
- matplotlib, plotly: visualization
- streamlit: web interface
- scikit-learn: analysis
- networkx: pathway graphs

## license

mit license

## references

chemokine receptor biology, signal transduction, and leukocyte migration modeling
