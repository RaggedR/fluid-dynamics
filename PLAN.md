# Plan: Vorticity-Streamfunction Fluid Solver with Evolving Parameters

## Context

Extending the reaction-diffusion playground concept to fluid dynamics. We start with a 2D PDE (vorticity-streamfunction formulation of incompressible Navier-Stokes), parameterize it with 5 values that control Kelvin-Helmholtz vortex formation, define "interestingness" as a fitness function, and evolve the parameters with an island-model GA. The project lives at `~/git/fluid-dynames/`.

The Kelvin-Helmholtz instability is the fluid-dynamics analogue of Turing patterns — a low-dimensional parameter space controls which qualitative morphology emerges.

## Architecture: Single HTML File

Following the `evolution.html` pattern from reaction-diffusion-playground: a self-contained HTML file with inline GLSL shaders, Three.js via CDN import map, no build tool. Zero-friction: open in browser and it works.

**Template file:** `/Users/robin/git/reaction-diffusion-playground/evolution.html`

## Key Design Decisions

| Question | Answer | Why |
|----------|--------|-----|
| Render targets | 4 (ω₀, ω₁, ψ₀, ψ₁) | Can't pack ω+ψ because Jacobi must update ψ while holding ω fixed |
| Advection | Semi-Lagrangian | Unconditionally stable, standard for GPU fluids, 1 pass per timestep |
| Poisson solver | 40 Jacobi iterations | Simple, warm-started from previous timestep, ~40 render passes |
| Genome | 5 params: ν, v_shear, δ, A_pert, k_pert | δ (shear layer thickness) dramatically changes visual character |
| Fitness | Shannon entropy of \|ω\| histogram + alive penalty | Direct port from reaction-diffusion; cheap on GPU, proven to work |
| Boundaries | Periodic everywhere (fract() wrapping) | Simplest; natural for KH in x, avoids wall treatment in y |
| Build tool | None — single HTML file | Matches evolution.html pattern, zero setup friction |

## The PDE

```
∂ω/∂t + (u·∇)ω = ν·∇²ω          (vorticity transport)
∇²ψ = -ω                          (Poisson equation for streamfunction)
u = ∂ψ/∂y,  v = -∂ψ/∂x           (velocity recovery)
```

## Genome (5 Parameters)

| Param | Role | Init range | Search range | Mutation σ |
|-------|------|-----------|--------------|-----------|
| ν | Viscosity (dissipation) | [0.0005, 0.005] | [0.0001, 0.01] | 0.001 |
| v_shear | Shear velocity | [1.0, 4.0] | [0.5, 5.0] | 0.3 |
| δ | Shear layer thickness | [0.02, 0.1] | [0.01, 0.15] | 0.01 |
| A_pert | Perturbation amplitude | [0.05, 0.3] | [0.01, 0.5] | 0.03 |
| k_pert | Perturbation wavenumber | [2, 6] | [1, 8] | 0.5 |

## 5 GLSL Shaders (all inline in HTML)

1. **VERT** — shared vertex shader, passes UV coordinates
2. **INIT_FRAG** — seeds KH initial condition: `ω = (v_shear/δ) · sech²((y-0.5)/δ) · (1 + A_pert · sin(2π·k_pert·x))`
3. **JACOBI_FRAG** — one Poisson iteration: `ψ_new = (ψ_neighbors + dx²·ω) / 4`
4. **ADVECT_FRAG** — Semi-Lagrangian advection + viscous diffusion of ω
5. **DISPLAY_FRAG** — diverging colormap (blue → white → red) from vorticity

## Simulation Loop (per timestep)

```
1. Poisson solve (40 Jacobi iterations):
   for j in 0..39:
     bind tOmega = omega[cur], tPsi = psi[j%2]
     render JACOBI_FRAG → psi[(j+1)%2]

2. Advection-diffusion:
   bind tOmega = omega[cur], tPsi = psi[converged]
   render ADVECT_FRAG → omega[1-cur]
   swap cur
```

## Implementation Phases

### Phase 1: Standalone fluid solver (no GA) — `fluid.html`
1. HTML skeleton with Three.js import map + canvas
2. 4 Float32 render targets (ω₀, ω₁, ψ₀, ψ₁)
3. VERT shader (passthrough UVs)
4. INIT_FRAG shader (KH initial condition)
5. JACOBI_FRAG shader (Poisson iteration)
6. ADVECT_FRAG shader (Semi-Lagrangian + diffusion)
7. DISPLAY_FRAG shader (diverging colormap)
8. Simulation loop + animation frame
9. Basic sliders for ν, v_shear, δ, A_pert, k_pert + reset button
10. **Verify**: KH vortices appear and roll up

### Phase 2: GA integration
1. Port Individual/Island/migration classes from evolution.html
2. Adapt genome to 5 fluid parameters
3. Simulator.evaluate() with fitness readback
4. Island grid UI, fitness chart, hall of fame
5. Topology selector + Evolve/Pause/Reset buttons
6. **Verify**: fitness increases, diverse vortex patterns evolve

### Phase 3: Batch experiments
1. Batch experiment UI panel + CSV export
2. Diversity metrics + charts
3. **Verify**: 4-topology comparison produces expected diversity dynamics

### Phase 4: CPU batch runner — `batch_fluid.mjs`
1. CPU Jacobi + Semi-Lagrangian + Laplacian
2. Port GA logic, same CSV output format
3. **Verify**: CPU/GPU fitness values are comparable for same parameters

## Numerical Defaults

```javascript
simSize: 128, simSteps: 300, jacobiIterations: 40,
stepsPerFrame: 4, dt: 0.05, omegaMax: 10.0
```

## Risk Mitigations

- **Blowup**: Clamp ω to [-omegaMax, omegaMax] in advection shader; blown-up sims get low fitness naturally
- **Jacobi convergence**: Warm-start from previous timestep; 40 iterations is sufficient at 128×128
- **Semi-Lagrangian diffusion**: Use small dt (0.05) to minimize numerical diffusion; increase simSteps to compensate

## Files to Create

| File | Description |
|------|-------------|
| `~/git/fluid-dynames/fluid.html` | Main application — solver + GA + UI |
| `~/git/fluid-dynames/batch_fluid.mjs` | CPU batch runner (Phase 4) |
| `~/git/fluid-dynames/APPROACHES.md` | Already created — design notes |

## Confidence

**85%** that Phase 1 produces visible KH vortices on first or second attempt. The Semi-Lagrangian + Jacobi approach is battle-tested in GPU fluid literature. The main uncertainty is tuning dt and Jacobi iterations for visual quality.
