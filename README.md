# Evolving Fluid Dynamics for Visual Interestingness

A genetic algorithm that evolves the parameters of a 2D Navier-Stokes fluid simulation to produce maximally "interesting" vortex patterns — where interestingness is measured as Shannon entropy of the vorticity histogram.

## The idea

Kelvin-Helmholtz instabilities are the fluid-dynamics analogue of Turing patterns in reaction-diffusion systems. A low-dimensional parameter space controls which qualitative morphology emerges: clean symmetric rolls, chaotic vortex merging, or full turbulence. This makes them ideal targets for evolutionary search.

An island-model GA searches a 5-parameter space controlling the initial conditions and viscosity of a vorticity-streamfunction Navier-Stokes solver. Fitness is Shannon entropy of the |omega| histogram — high entropy means the simulation produces a wide range of vorticity magnitudes (complex flow structures) rather than converging to a trivial steady state.

## The PDE

```
dw/dt + (u . grad)w = nu * laplacian(w)     (vorticity transport)
laplacian(psi) = -w                          (Poisson equation for streamfunction)
u = dpsi/dy,  v = -dpsi/dx                  (velocity recovery)
```

## The genome

| Parameter | Role | Search range |
|-----------|------|-------------|
| nu | Viscosity (dissipation rate) | [0.0001, 0.01] |
| v_shear | Shear velocity at interface | [0.5, 5.0] |
| delta | Shear layer thickness | [0.01, 0.15] |
| A_pert | Perturbation amplitude | [0.01, 0.5] |
| k_pert | Perturbation wavenumber | [1, 8] |

These parameters control the Reynolds number and initial instability structure. Small changes produce dramatically different vortex evolution.

## Running

**Interactive solver** — open `fluid.html` in a browser. Tweak parameters with sliders and watch Kelvin-Helmholtz vortices form in real time.

**GA evolution** — open `evolution.html` in a browser. An island-model GA evolves fluid parameters, with a fitness chart and hall of fame showing the most interesting simulations found so far.

**Headless batch** — `node batch_fluid.mjs` runs experiments on CPU without a browser, outputting results to CSV.

## Implementation

Everything runs on the GPU via WebGL/Three.js with inline GLSL shaders:

- 4 render targets (omega_0, omega_1, psi_0, psi_1) with ping-pong buffering
- 40 Jacobi iterations per timestep for the Poisson solve
- Semi-Lagrangian advection for unconditional stability
- Diverging blue-white-red colormap for vorticity visualization

Self-contained HTML files with no build step — open in a browser and it works.

## Design notes

See [APPROACHES.md](APPROACHES.md) for a comparison of four candidate PDE formulations (vorticity-streamfunction, lattice Boltzmann, shallow water, Navier-Stokes-Korteweg) and why vorticity-streamfunction was chosen as the starting point. See [PLAN.md](PLAN.md) for the phased implementation plan.

## Lineage

Extends the [reaction-diffusion-playground](https://github.com/RaggedR/reaction-diffusion-playground) (itself a fork of [jasonwebb/reaction-diffusion-playground](https://github.com/jasonwebb/reaction-diffusion-playground)), replacing Gray-Scott reaction-diffusion with Navier-Stokes fluid dynamics while reusing the same island-model GA and fitness framework.
