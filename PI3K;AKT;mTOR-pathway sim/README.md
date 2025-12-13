# pi3k / akt / mtor pathway (ode research utility)

slug: `pi3k_akt_mtor`

this repository implements an ode-style, toggleable-variant model of pi3k–akt–mtor signaling for exploratory research workflows (survival, metabolism, growth; cancer-relevant feedback/crosstalk).

design goals:
- explicit, inspectable equations (no hidden ml components)
- modular variants (baseline, feedback, crosstalk, feedback+crosstalk)
- reproducible simulation/export (timeseries + metadata)

## quick model summary

baseline signal flow (normalized activities):

rtk (`R`) → pi3k (`P`) → pip3 (`PIP3`) → akt (`Ap`) → tsc gate (`Ti`) → mtorc1 (`M1`) → s6k (`S`)

additional branches:
- mtorc2 (`M2`) promotes akt phosphorylation
- foxo (`F`) is akt-inhibited and can restore irs1 function (adaptive response)
- erk (`X`) and ampk (`K`) are optional inputs for crosstalk
- autophagy (`Au`) increases with ampk and decreases with mtorc1

## state variables (dimensionless, clamped to [0, 1])

all model states are fractions/activities in $[0,1]$. the solver may transiently step outside; downstream rates use clamped states.

primary states:
- `R`: active receptor/rtk
- `Is`: inhibited irs1 fraction (functional irs1 is `I = 1 - Is`)
- `P`: active pi3k
- `PIP3`: pip3 fraction (pip2 is `1 - PIP3`)
- `Ap`: active/phosphorylated akt
- `F`: active foxo (akt-inhibited; default initial value is 1.0)
- `Ti`: inhibited tsc fraction (active tsc is `1 - Ti`)
- `M1`: active mtorc1
- `S`: active s6k
- `M2`: active mtorc2
- `X`: active erk (optional)
- `K`: active ampk (optional)
- `Au`: autophagy activity

## external inputs and drug multipliers

inputs are in `Inputs`:
- `ligand` in $[0,1]$: upstream rtk drive
- `erk_input` in $[0,1]$: erk drive (for crosstalk experiments)
- `ampk_input` in $[0,1]$: ampk drive (for energy-stress experiments)

drug multipliers (all in $[0,1]$; 1.0 = no inhibition, 0.0 = full inhibition):
- `pi3k_activity`
- `akt_activity`
- `mtorc1_activity`

## derived readouts (exported by default)

the simulator appends simple, transparent composites for research-facing visualization:

- `IRS1_func` = $1 - Is$
- `TSC_act` = $1 - Ti$
- `Growth` (proxy) = $\mathrm{clip}(0.6\,M1 + 0.4\,S)$
- `Survival` (proxy) = $\mathrm{clip}(0.7\,Ap + 0.3\,M2 - 0.3\,Au)$
- `Metabolism` (proxy) = $\mathrm{clip}(0.5\,Ap + 0.5\,M1)$

these are not claims of quantitative phenotype; they are convenience readouts for rapid hypothesis exploration.

## ode system (matches code parameter names)

let all states be functions of time $t$; `clip` means clamping to $[0,1]$.

define:
- $I = 1 - Is$ (functional irs1)
- $PIP2 = 1 - PIP3$

receptor:
$$\frac{dR}{dt} = k\_R\_on \cdot ligand \cdot (1-R) - k\_R\_off \cdot R$$

erk and ampk (driven by external inputs):
$$\frac{dX}{dt} = k\_{ERK\_on} \cdot erk\_input \cdot (1-X) - k\_{ERK\_off} \cdot X$$
$$\frac{dK}{dt} = k\_{AMPK\_on} \cdot ampk\_input \cdot (1-K) - k\_{AMPK\_off} \cdot K$$

irs1 inhibition/recovery (feedback + rescue):
$$inhib\_drive = k\_{S6K\_to\_IRS} \cdot S + k\_{ERK\_to\_IRS} \cdot X$$
$$rescue\_drive = k\_{FOXO\_to\_IRS\_rescue} \cdot F$$
$$\frac{dIs}{dt} = inhib\_drive \cdot (1-Is) - (k\_{IRS\_deinhib} + rescue\_drive) \cdot Is$$

pi3k:
$$\frac{dP}{dt} = pi3k\_activity \cdot (k\_{PI3K\_on} \cdot R \cdot I \cdot (1-P)) - k\_{PI3K\_off} \cdot P$$

pip3 balance (pi3k vs pten-like removal):
$$\frac{dPIP3}{dt} = pi3k\_activity \cdot (v\_{PI3K} \cdot P \cdot PIP2) - v\_{PTEN} \cdot PIP3$$

mtorc2 with s6k negative feedback:
$$\frac{dM2}{dt} = k\_{mTORC2\_on} \cdot R \cdot (1-M2) - (k\_{mTORC2\_off} + k\_{S6K\_to\_mTORC2\_inhib} \cdot S) \cdot M2$$

akt phosphorylation from pip3 and mtorc2:
$$\frac{dAp}{dt} = akt\_activity \cdot ((v\_{PDK1} \cdot PIP3 + v\_{mTORC2\_to\_AKT} \cdot M2) \cdot (1-Ap)) - v\_{AKT\_dephos} \cdot Ap$$

foxo (akt-inhibited):
$$\frac{dF}{dt} = k\_{FOXO\_on} \cdot (1-Ap) \cdot (1-F) - (k\_{FOXO\_off} + k\_{AKT\_to\_FOXO\_inhib} \cdot Ap) \cdot F$$

tsc inhibition/recovery (akt + erk):
$$\frac{dTi}{dt} = (k\_{AKT\_to\_TSC} \cdot Ap + k\_{ERK\_to\_TSC} \cdot X) \cdot (1-Ti) - k\_{TSC\_recover} \cdot Ti$$

mtorc1 activation gated by active tsc and inhibited by ampk:
$$ampk\_gate = \max(0, 1 - alpha\_{AMPK} \cdot K)$$
$$\frac{dM1}{dt} = mtorc1\_activity \cdot (k\_{mTORC1\_on} \cdot (1-Ti) \cdot ampk\_gate \cdot (1-M1)) - k\_{mTORC1\_off} \cdot M1$$

s6k:
$$\frac{dS}{dt} = k\_{S6K\_on} \cdot M1 \cdot (1-S) - k\_{S6K\_off} \cdot S$$

autophagy (ampk-promoted, mtorc1-suppressed):
$$aut\_drive = \max(0, beta\_{AUT\_AMPK} \cdot K - beta\_{AUT\_mTORC1} \cdot M1)$$
$$\frac{dAu}{dt} = k\_{AUT\_on} \cdot aut\_drive \cdot (1-Au) - k\_{AUT\_off} \cdot Au$$

## variants (feedback and crosstalk)

variants are implemented by modifying parameters (see `apply_variant`):

- `baseline`: no additional toggles
- `feedback`:
  - increases `k_S6K_to_IRS` (s6k ⟂ irs1)
- `crosstalk`:
  - increases `k_ERK_to_TSC` (erk ⟂ tsc)
  - increases `k_ERK_to_IRS` (erk ⟂ irs1)
  - increases `alpha_AMPK` (ampk ⟂ mtorc1)
- `feedback_crosstalk`: combines both

## numerical details

- solver: `scipy.integrate.solve_ivp(method="LSODA")`
- default tolerances: `rtol=1e-6`, `atol=1e-9`
- time units: arbitrary (interpret relative timescales, not absolute minutes, unless you calibrate parameters)

## run

recommended (virtualenv in this repo):

### cli

- `pip install -e .`
- `pi3k-akt-mtor --help`
- example (export timeseries + metadata):
  - `pi3k-akt-mtor --variant feedback --t-end 120 --ligand 1.0 --export out.csv --export-meta out.json`
- export only raw state columns (no derived):
  - `pi3k-akt-mtor --no-derived --export out.csv`

### interactive ui (streamlit)

- `python -m streamlit run app.py`

if you accidentally run `streamlit` from another python install, you may see `ModuleNotFoundError: pi3k_akt_mtor`. running it as `python -m streamlit ...` forces the correct interpreter.

## interpretation and scope

- this is a mechanistic, reduced model intended for controlled exploration, sensitivity checks, and "what-if" perturbations.
- it is not a curated biochemistry database and does not encode isoform-specific details (pi3k isoforms, akt1/2/3, mtorc1 substrates beyond s6k, etc.) unless you add them.
