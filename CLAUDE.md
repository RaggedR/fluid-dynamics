# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Extending the reaction-diffusion playground to evolve fluid dynamics PDEs. Uses vorticity-streamfunction formulation of incompressible Navier-Stokes to evolve Kelvin-Helmholtz vortex patterns. An island-model GA searches a 5-parameter space to maximize "visual interestingness" (Shannon entropy of vorticity histogram).

## Stack

- WebGL/Three.js for GPU fluid simulation (GLSL shaders)
- JavaScript/HTML
- Node.js batch runner (batch_fluid.mjs)

## Architecture

- `fluid.html` — Phase 1: standalone solver with UI sliders
- `evolution.html` — Phase 2 reference: full GA with island topology, fitness chart, hall of fame
- `batch_fluid.mjs` — Phase 4: CPU batch runner for headless experiments
- `APPROACHES.md` — Design decisions comparing LBM vs vorticity-streamfunction
- `PLAN.md` — Phase breakdown with numerical defaults and risk mitigations

## Key Design

- 4 render targets: ω₀, ω₁, ψ₀, ψ₁ (can't pack because Jacobi must update ψ while holding ω)
- 40 Jacobi iterations per timestep for Poisson solve
- 5 genome parameters: ν (viscosity), v_shear, δ (layer thickness), A_pert, k_pert
- Fitness: Shannon entropy of |ω| histogram
- Related: `reaction-diffusion-playground/` is the parent project
