# Evolving Fluid Dynamics PDEs for "Interesting" Swirls

Extending the reaction-diffusion playground concept: start with a PDE, parameterize it,
define a fitness function for visual interestingness, and evolve the parameters with an
island-model GA.

The domain is 2D fluid dynamics — thin-film / petri-dish flows that produce
Kelvin-Helmholtz vortices, spirals, and turbulent mixing patterns.

---

## Approach 1: Lattice Boltzmann Method (LBM)

**What it is**: Not a PDE but a mesoscopic particle model. Fluid is represented as
distribution functions on a discrete velocity lattice (D2Q9 — 9 velocities per cell).
Each timestep: (1) stream distributions to neighbors, (2) collide locally via BGK
relaxation.

**The equation** (BGK collision):
```
f_i(x + e_i·dt, t + dt) = f_i(x, t) - (1/τ)(f_i - f_i^eq)
```
where `f_i^eq` is the Maxwell-Boltzmann equilibrium and `τ` is the relaxation time.

**Evolvable parameters** (genome):
| Parameter | Role | Typical range |
|-----------|------|---------------|
| τ (relaxation time) | Controls viscosity: ν = (τ - 0.5)/3 | [0.51, 2.0] |
| v_shear | Initial shear velocity between layers | [0.01, 0.3] |
| ρ_ratio | Density ratio between top/bottom fluid | [0.5, 2.0] |
| perturbation_k | Wavenumber of initial sinusoidal perturbation | [1, 16] |

**Pros**:
- Embarrassingly parallel — each cell is a local update, perfect for GPU shaders
- Very similar architecture to reaction-diffusion ping-pong buffers
- Naturally handles complex boundaries
- Stable without special treatment

**Cons**:
- Not a "real PDE" — harder to frame in the GECCO paper narrative
- Requires 9 floats per cell (vs 2 for reaction-diffusion), so heavier on texture memory
- Compressibility artifacts at high velocities

**Implementation notes**:
- D2Q9 lattice fits in a 3×3 stencil (same as Gray-Scott Laplacian)
- Could pack 9 distribution values across 3 RGBA textures (3 ping-pong pairs)
- Or use WebGL2 with larger texture formats

---

## Approach 2: Vorticity-Streamfunction Formulation ← START HERE

**What it is**: The 2D incompressible Navier-Stokes equations reduced to a single scalar
PDE by working in vorticity (ω) instead of velocity (u, v). Eliminates pressure entirely.
This is the natural formulation for thin-film / petri-dish flows.

**The equations**:
```
∂ω/∂t + (u·∇)ω = ν·∇²ω          (vorticity transport)
∇²ψ = -ω                          (Poisson equation for streamfunction)
u = ∂ψ/∂y,  v = -∂ψ/∂x           (velocity recovery)
```

where:
- ω = vorticity (curl of velocity) — the "swirliness" at each point
- ψ = streamfunction — contours are flow lines
- ν = kinematic viscosity — how fast vortices dissipate
- The advection term (u·∇)ω is what creates the Kelvin-Helmholtz rolls

**Evolvable parameters** (genome):
| Parameter | Role | Typical range |
|-----------|------|---------------|
| ν (viscosity) | Dissipation rate; low ν = turbulent, high ν = laminar | [0.0001, 0.01] |
| v_shear | Shear velocity magnitude at the interface | [0.01, 0.5] |
| A_pert | Amplitude of initial perturbation | [0.001, 0.1] |
| k_pert | Wavenumber of initial perturbation (how many rolls) | [1, 16] |

These four parameters control the Reynolds number (Re = v_shear · L / ν) and the
initial instability structure. Small changes produce dramatically different vortex
evolution — from clean symmetric rolls to chaotic turbulent mixing.

**Key analogy**: Kelvin-Helmholtz instabilities are the fluid-dynamics analogue of
Turing patterns in reaction-diffusion. Just as Gray-Scott's f/k ratio determines
spots vs stripes vs labyrinths, the Reynolds number (Re = v_shear · L / ν) determines
whether you get clean symmetric rolls, chaotic vortex merging, or full turbulence.
Both are examples of a low-dimensional parameter space controlling which qualitative
morphology emerges from a PDE — making them ideal targets for evolutionary search.

**Why this is the right starting point**:
1. Single scalar PDE on a 2D grid — structurally identical to Gray-Scott
2. Directly produces Kelvin-Helmholtz vortices — the "swirly smoke" patterns
3. Four parameters map naturally to a GA genome
4. Well-studied physics — we know what parameter ranges produce interesting behavior
5. Fits the GECCO narrative: "evolving PDE parameters for emergent complexity"

**Implementation plan**:
1. **Vorticity transport**: Advection-diffusion on ω, solved with finite differences
   - Advection: upwind or semi-Lagrangian scheme for stability
   - Diffusion: standard 5-point Laplacian (reuse from Gray-Scott)
2. **Poisson solve for ψ**: Iterative Jacobi relaxation on GPU
   - Run ~50-100 Jacobi iterations per timestep inside the shader
   - Or use a multi-grid approach for faster convergence
3. **Velocity recovery**: Central differences on ψ to get (u, v)
4. **Rendering**: Map vorticity magnitude to color (similar to Gray-Scott B channel)

**Shader architecture** (extends existing ping-pong pattern):
- Texture A: ω (vorticity) — ping-pong between two render targets
- Texture B: ψ (streamfunction) — solved iteratively each frame
- Pass 1: Poisson solve (many sub-iterations) → updates ψ from current ω
- Pass 2: Advect + diffuse ω using velocities derived from ψ

**Initial conditions** (Kelvin-Helmholtz setup):
- Top half: velocity = +v_shear (rightward)
- Bottom half: velocity = -v_shear (leftward)
- Interface: thin shear layer with sinusoidal perturbation
  - ω_init(x, y) = (v_shear / δ) · sech²((y - 0.5) / δ) · (1 + A_pert · sin(2π · k_pert · x))
  - δ = shear layer thickness (could be a 5th parameter, or fixed)

**Boundary conditions**:
- Periodic in x (flow wraps around)
- Free-slip (ω = 0) or periodic in y

---

## Approach 3: Shallow Water Equations

**What it is**: Depth-averaged 2D fluid equations. Models waves and flows in a thin
layer of fluid under gravity.

**The equations**:
```
∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0              (mass conservation)
∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = 0  (x-momentum)
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = 0  (y-momentum)
```

**Evolvable parameters**:
| Parameter | Role | Typical range |
|-----------|------|---------------|
| g (gravity) | Wave speed scaling | [0.1, 10.0] |
| friction | Bottom drag coefficient | [0.0, 0.1] |
| h_drop | Initial height perturbation amplitude | [0.01, 1.0] |
| drop_radius | Spatial scale of perturbation | [0.01, 0.3] |

**Character**: More wave-like than swirly. Produces expanding ring waves, reflections,
interference patterns. Less Kelvin-Helmholtz, more "stone dropped in pond." Could be
interesting for evolving wave interference patterns rather than vortices.

**Pros**: Simple hyperbolic system, well-suited to GPU, dramatic visuals
**Cons**: Fewer vortices — not the "swirly smoke" aesthetic we want first

---

## Approach 4: Navier-Stokes-Korteweg (Diffuse Interface)

**What it is**: Navier-Stokes augmented with a capillary stress tensor from phase-field
theory. Models two immiscible fluids with a diffuse interface — no need to track a sharp
boundary.

**The equations** (simplified 2D):
```
∂ρ/∂t + ∇·(ρu) = 0
∂(ρu)/∂t + ∇·(ρu⊗u) = -∇p + μ∇²u + κρ∇(∇²ρ)
```

The key term is the Korteweg stress `κρ∇(∇²ρ)` — it creates surface tension effects
at the interface between two fluid densities.

**Evolvable parameters**:
| Parameter | Role | Typical range |
|-----------|------|---------------|
| μ (viscosity) | Dissipation | [0.001, 0.1] |
| κ (capillarity) | Surface tension strength | [0.0001, 0.01] |
| ρ_ratio | Density ratio of the two fluids | [0.1, 10.0] |
| v_shear | Initial shear velocity | [0.01, 0.5] |

**Character**: Produces beautiful interfacial spirals — like oil and water being stirred.
The capillary term creates fine filaments and droplet breakup that pure Navier-Stokes
doesn't capture.

**Pros**: Stunning visuals, physically rich behavior
**Cons**: More complex (coupled density + momentum), requires careful numerics, κ term
involves 4th-order spatial derivatives

---

## Fitness Function: "Interestingness of Swirls"

Adapting the reaction-diffusion entropy approach to vortex fields:

### Candidate metrics

1. **Enstrophy spectrum entropy**
   - Compute enstrophy (∫ω² dA) at different spatial scales via coarse-graining
   - Take Shannon entropy of the scale distribution
   - High entropy = vortices at many scales = visually complex

2. **Vortex count and size distribution**
   - Threshold |ω| to identify vortex cores
   - Count distinct vortices and measure their radii
   - Reward: many vortices of varied sizes (power-law-ish distribution)

3. **Temporal persistence**
   - Measure how the vorticity field changes between timesteps
   - Too static = dead; too chaotic = noise; moderate change = dynamic patterns
   - Use autocorrelation of ω field across time

4. **Spatial structure (2-point correlation)**
   - Compute radial correlation function of ω
   - Reward long-range correlations (organized structure) over white noise

### Recommended composite fitness
```
fitness = spectral_entropy × persistence_score × (0.3 + 0.7 × alive_penalty)
```

Where `alive_penalty = 4r(1-r)` with r = fraction of cells with |ω| above threshold.
This mirrors the Gray-Scott approach while capturing fluid-specific structure.

---

## Implementation Order

1. **Approach 2** (vorticity-streamfunction) — closest to existing architecture, produces
   the target aesthetic, clean 4-parameter genome
2. **Approach 1** (LBM) — natural second step, different numerical method but same
   evolutionary framework, richer dynamics
3. **Approach 4** (Navier-Stokes-Korteweg) — most visually stunning but most complex
4. **Approach 3** (shallow water) — different aesthetic (waves not swirls), worth
   exploring as a contrast

Each approach reuses the same GA infrastructure (island model, migration topologies,
batch experiments) from reaction-diffusion-playground. Only the simulation kernel and
fitness function change.
