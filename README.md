# biosim signaling pathway sims

## overview

this repository is a minimal, modular library of signaling pathway simulations. each module captures a canonical pathway as a standalone simulation scaffold that can be reused, composed, or extended.

the goal is **infrastructure**, not finished models: a clean starting point for building larger, disease- or context-specific systems.

## scope

current modules cover a wide slice of cell signaling, including:

- developmental & morphogen pathways  
  - hedgehog pathway sim  
  - wnt : β-catenin pathway sim  
  - notch pathway sim  
  - hippo : yap-taz pathway sim  

- growth factor & rtk networks  
  - rtk:egfr signaling sim  
  - fgf:fgfr signaling sim  
  - hgf:met signaling sim  
  - pdgf signaling sim  
  - vegf signaling sim  
  - c-kit : stem cell factor signaling sim  
  - neurotrophin (ngf:trka) signaling sim  
  - insulin:igf signaling sim  

- metabolic & stress sensing  
  - ampk energy-sensing pathway sim  
  - pi3k : akt : mtor pathway sim  

- nuclear & hormone receptors  
  - androgen receptor signaling sim  
  - estrogen receptor signaling sim  
  - progesterone receptor signaling sim  
  - thyroid hormone receptor signaling sim  
  - ppar-α-γ-δ signaling sim  

- immune & inflammatory signaling  
  - nf-κb pathway sim  
  - type i interferon (ifn-α/β) signaling sim  
  - type ii interferon (ifn-γ) signaling sim  
  - t-cell receptor signaling sim  
  - b-cell receptor signaling sim  
  - nlrp3 inflammasome activation sim  
  - complement cascade signaling sim  
  - chemokine receptor cxcr/ccr signaling sim  
  - fc receptor (fcγr) signaling in innate cells sim  

- core intracellular signaling axes  
  - jak-stat pathway sim  
  - mapk : erk pathway sim  
  - gpcr : camp : pka pathway sim  
  - calcium signaling sim  
  - integrin : fak signaling sim  
  - rho gtpase : rho-rock pathway sim  
  - eph : ephrin signaling sim  

each module is designed to be:

- **minimal** – only the essential species, interactions, and parameters  
- **composable** – can be coupled to other pathways without heavy refactoring  
- **transparent** – clear structure, explicit assumptions, and modifiable defaults  

## usage

intended usage patterns:

- use a single pathway module as a **standalone simulation** (for exploration, teaching, or rapid hypothesis checks).  
- combine multiple modules into a **larger network** for disease or cell-type specific models.  
- swap or refine submodules (e.g., replace a simple nf-κb model with a more detailed one) while keeping the surrounding architecture intact.  

see individual pathway folders/files for:

- model structure and assumptions  
- parameters, initial conditions, and outputs  
- notes on typical use cases and limitations  

## motivation

this repo exists as **plumbing** for future work:

- to avoid rebuilding the same canonical pathways from scratch for every project  
- to have a consistent, modular base when constructing larger models (e.g., disease-focused systems, organ-level frameworks, or multi-omic digital twins)  
- to make it easy to test different combinations and couplings of pathways without fighting with ad-hoc code each time  

in short: this is a reusable biosim backbone for a specific downstream modeling program, keeping the low-level signaling mechanics clean, modular, and ready to plug into higher-level projects.
