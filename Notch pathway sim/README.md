# notch

cell–cell communication for fate decisions and patterning.

a compact simulation kit for notch–delta lateral inhibition with two model styles:

- ode: continuous multicellular model on a 2d lattice
- boolean: logical update model for fast intuition

## quickstart

requirements: python 3.10+

install (editable):

- `python -m pip install -e .`

run ode demo (shows salt-and-pepper patterns for many parameter settings):

- `notch ode --rows 20 --cols 20 --t 40 --dt 0.2 --plot`

run boolean demo:

- `notch boolean --rows 30 --cols 30 --steps 80 --noise 1.0 --plot`

save outputs:

- `notch ode --save-npz out_ode.npz`
- `notch boolean --save-npz out_bool.npz`

## streamlit

install ui extra:

- `python -m pip install -e ".[ui]"`

run:

- `streamlit run notch/streamlit_app.py`

usage:

- open the local url streamlit prints
- click `run` once, then adjust sliders (it auto-reruns)
- optionally upload an edge list to override grid adjacency

run on an irregular contact graph (edge list):

- `notch ode --rows 2 --cols 3 --edges examples/edges_small.txt --t 20 --dt 0.2`

edge list notes:

- 0-based node indices
- for a `rows x cols` grid, node id is `r*cols + c`

minimal sweep (csv to stdout):

- `notch sweep --rows 20 --cols 20 --alpha 1,2,4,8 --t 40 --dt 0.2`

write sweep to a file:

- `notch sweep --rows 20 --cols 20 --alpha 1,2,4,8 --t 40 --dt 0.2 --out sweep.csv`

## model notes

ode state per cell $i$:

- $n_i$: notch receptor level
- $d_i$: delta ligand level
- $icd_i$: nicd / notch activity

coupling: each cell senses neighbor delta mean $\bar d_i$ on a grid adjacency (von neumann or moore).

a typical form (this repo’s implementation):

- $\dot n_i = p_n + \beta\,h_{act}(icd_i) - g_n n_i$
- $\dot d_i = p_d\,h_{rep}(icd_i) - g_d d_i$
- $\dot {icd}_i = \alpha\,n_i\,h_{act}(\bar d_i) - g_i icd_i$

with hill functions:

- $h_{act}(x)=\frac{x^n}{k^n+x^n}$
- $h_{rep}(x)=1-h_{act}(x)$

boolean model:

- $icd_i \leftarrow 1$ if neighbor-delta mean exceeds a threshold
- $d_i \leftarrow 1$ if $icd_i=0$ and a (biased + noisy) intrinsic drive is positive

## files

- [notch/ode.py](notch/ode.py): ode model + solver
- [notch/boolean.py](notch/boolean.py): boolean model
- [notch/neighbors.py](notch/neighbors.py): grid adjacency + neighbor averaging
- [notch/cli.py](notch/cli.py): `notch` command

## common tweaks

- stronger lateral inhibition: increase `alpha`, increase `n_rep`, decrease `k_rep`
- more spatial coupling: switch to `--topology moore`
- reduce symmetry: increase `eps` slightly
