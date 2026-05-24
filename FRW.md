# Floating Random Walk Capacitance Solver

This note records the intended Floating Random Walk (FRW) solver design for
`capext`. The first target is a homogeneous dielectric with axis-aligned cuboid
conductors, matching the current BEM problem model.

## Why the prototype can still differ from BEM

The current `frw.py` prototype has the right solver skeleton:

- it starts from a Gaussian surface around the observation conductor;
- it uses the largest conductor-free transition cube by default;
- it samples a centered-cube surface Green-function PDF for potential walks;
- it applies the first-step \(\omega\) weight from the normal derivative of the
  surface Green function;
- it terminates on conductors or on an artificial outer reference box;
- it records walk statistics and representative walk traces.

The ordinary potential walk uses the surface Green function as a transition
PDF. Capacitance, however, is a flux integral, so the first transition from the
Gaussian surface must use the normal derivative of that Green function.
QuickCap, RWCap, and the older FRW thesis all use this field/weight
formulation.

The implementation is still a research prototype. Differences from BEM can
come from stochastic variance, the finite transition-cube table, the simple
box-shaped Gaussian surface, and missing RWCap production features such as
variance reduction and space-management acceleration.

## Physical problem

For a homogeneous dielectric, the potential in the dielectric region satisfies

$$
\nabla^2 \phi = 0.
$$

Each conductor net is an equipotential Dirichlet boundary. For the \(k\)-th
capacitance-column experiment,

$$
\phi^{(k)} = 1 \quad \text{on conductor } k,
$$

and

$$
\phi^{(k)} = 0 \quad \text{on all other conductors and the reference boundary}.
$$

The Maxwell capacitance coefficient is the charge on observation conductor
\(i\):

$$
C_{ik}=Q_i^{(k)}
=-\int_{\Gamma_i}\epsilon
\frac{\partial \phi^{(k)}}{\partial n}\,dS,
$$

where \(\Gamma_i\) is the conductor surface and \(n\) follows the sign
convention used by the capacitance formulation.

In FRW we normally evaluate the same flux on a Gaussian surface \(G_i\) that
encloses conductor \(i\) and no other conductor:

$$
C_{ik}=Q_i^{(k)}
=-\int_{G_i}\epsilon
\frac{\partial \phi^{(k)}}{\partial n}\,dS.
$$

## Potential Green-function identity

Let \(S\) be a transition-domain boundary around an interior point
\(\mathbf r\). The FRW potential walk uses the surface Green function
\(P(\mathbf r,\mathbf r^{(1)})\):

$$
\phi(\mathbf r)
=
\oint_S P(\mathbf r,\mathbf r^{(1)})
\phi(\mathbf r^{(1)})\,dS_{\mathbf r^{(1)}} .
$$

For fixed \(\mathbf r\), \(P\) is a PDF over \(S\):

$$
\oint_S P(\mathbf r,\mathbf r^{(1)})\,dS_{\mathbf r^{(1)}}=1.
$$

Therefore, after the first capacitance/field step has been handled, a potential
walk can sample \(\mathbf r^{(1)}\sim P(\mathbf r,\cdot)\) and continue without
an additional multiplicative weight for a homogeneous centered transition
domain.

## Deriving the capacitance weight

Start from the Gaussian-surface charge integral for excitation \(k\):

$$
Q_i^{(k)}
=
-\int_{G_i}\epsilon(\mathbf r)
\frac{\partial \phi^{(k)}(\mathbf r)}{\partial n}\,dS_{\mathbf r}.
$$

Insert the potential Green-function identity and differentiate with respect to
the source point \(\mathbf r\):

$$
\frac{\partial \phi^{(k)}(\mathbf r)}{\partial n}
=
\oint_S
\nabla_{\mathbf r}P(\mathbf r,\mathbf r^{(1)})
\cdot \hat{\mathbf n}(\mathbf r)\,
\phi^{(k)}(\mathbf r^{(1)})
\,dS_{\mathbf r^{(1)}} .
$$

Then

$$
Q_i^{(k)}
=
\int_{G_i}\epsilon(\mathbf r)
\oint_S
\left[
-\nabla_{\mathbf r}P(\mathbf r,\mathbf r^{(1)})
\cdot \hat{\mathbf n}(\mathbf r)
\right]
\phi^{(k)}(\mathbf r^{(1)})
\,dS_{\mathbf r^{(1)}}\,dS_{\mathbf r}.
$$

RWCap introduces a surface normalization function \(g\) on the Gaussian surface
such that

$$
\int_{G_i}\epsilon(\mathbf r)g(\mathbf r)\,dS_{\mathbf r}=1.
$$

For homogeneous \(\epsilon\) and uniform Gaussian-surface sampling, this can be
chosen as

$$
g=\frac{1}{\epsilon A_{G_i}},
$$

where \(A_{G_i}\) is the area of the Gaussian surface. Multiplying and dividing
the inner integral by \(g(\mathbf r)P(\mathbf r,\mathbf r^{(1)})\) gives the
Monte Carlo form

$$
Q_i^{(k)}
=
\mathbb E\left[
\omega(\mathbf r,\mathbf r^{(1)})
\phi^{(k)}(\mathbf r^{(1)})
\right],
$$

where \(\mathbf r\) is sampled from the Gaussian-surface density
\(\epsilon g\), \(\mathbf r^{(1)}\) is sampled from
\(P(\mathbf r,\cdot)\), and the capacitance weight is

$$
\boxed{
\omega(\mathbf r,\mathbf r^{(1)})
=
-\frac{
\nabla_{\mathbf r}P(\mathbf r,\mathbf r^{(1)})
\cdot \hat{\mathbf n}(\mathbf r)
}{
g(\mathbf r)P(\mathbf r,\mathbf r^{(1)})
}
}.
$$

This is the \(\omega\) formula used in the RWCap derivation. QuickCap states
the same idea in electric-field form: after choosing a point on the Gaussian
surface, the first random step estimates the normal electric field using the
normal component of the electric-field Green function, equivalently the normal
derivative of the surface Green function.

After \(\mathbf r^{(1)}\) is sampled, the rest of the walk estimates
\(\phi^{(k)}(\mathbf r^{(1)})\). With conductor boundary values 0 or 1,

$$
\phi^{(k)}(\mathbf r^{(1)})
=
\Pr\{\text{potential walk from }\mathbf r^{(1)}
\text{ hits conductor }k\}.
$$

So one completed walk contributes

$$
\omega(\mathbf r,\mathbf r^{(1)})\,\mathbf e_h
$$

to the row for observation conductor \(i\), where \(h\) is the conductor hit by
the continuation walk. If the walk reaches the outer reference boundary, the
contribution is zero because the reference boundary is held at \(0\ \text{V}\).

This replaces the earlier prototype contribution

$$
\frac{\epsilon A_G}{d_G}(\mathbf e_i-\mathbf e_h),
$$

which was only a finite-distance flux approximation.

## Transition domains

The key acceleration in FRW is avoiding small grid steps. From a current point
\(\mathbf x\), construct a transition domain \(D(\mathbf x)\) that is fully in
the dielectric and touches no conductor. Then sample the next point on
\(\partial D\) according to the harmonic measure.

Common choices are:

- sphere: simple for homogeneous free space, connected to walk-on-spheres;
- cube: natural for Manhattan / rectilinear VLSI geometry;
- precharacterized transition cube: transition probabilities and weights are
  tabulated for reuse.

The first `capext` FRW implementation uses the largest cubic transition domain
centered at the current random-walk point.

At random-walk point \(\mathbf x\), define

$$
D(\mathbf x,a)
=
\left\{
\mathbf y:\ \lVert \mathbf y-\mathbf x\rVert_\infty < a
\right\},
$$

where

$$
a=s\,d_\infty(\mathbf x).
$$

Here \(s=\texttt{transition\_safety}\), and \(d_\infty(\mathbf x)\) is the
minimum \(L_\infty\) distance from \(\mathbf x\) to any conductor or the FRW
outer boundary. By default,

$$
s=1,
$$

so the code uses the maximum conductor-free transition cube, matching the
QuickCap/RWCap description. Smaller values of \(s\) are only a
debugging/robustness knob.

## Centered-cube potential PDF used in code

The current `frw.py` implementation uses the single-dielectric surface Green
function for a cube centered at the current random-walk point. Using a
normalized unit cube \([0,1]^3\), source point at the center
\((1/2,1/2,1/2)\), and the top face \(z=1\), the density used for one face is

$$
p_{\text{top}}(x,y)
=
\frac{4}{L^2}
\sum_{n_x=1}^{\infty}
\sum_{n_y=1}^{\infty}
\sin\left(\frac{\pi n_x}{2}\right)
\sin\left(\frac{\pi n_y}{2}\right)
\frac{\sinh(\pi n_z/2)}{\sinh(\pi n_z)}
\sin(\pi n_x x)
\sin(\pi n_y y),
$$

with

$$
n_z=\sqrt{n_x^2+n_y^2}, \qquad L=1.
$$

Equivalent formulas apply to the other five faces by symmetry. Since the walk
point is the cube center, each face integrates to \(1/6\):

$$
\int_0^1\int_0^1 p_{\text{top}}(x,y)\,dx\,dy
=
\frac{1}{6}.
$$

In code, `CenteredCubeGreenSampler` discretizes each face into an
\(N\times N\) table. The probability of cell \((i,j)\) on one face is
approximated by

$$
\Pr(i,j,\text{face})
\approx
p_{\text{top}}(x_i,y_j)\Delta x\Delta y,
\qquad
\Delta x=\Delta y=\frac{1}{N},
$$

then the table is normalized over all six faces. After a cell is selected, the
actual point is sampled uniformly inside that small cell and mapped from the
unit face to the physical cube face.

This is very different from a uniform cube-surface approximation

$$
p_{\text{uniform}}(\mathbf y\mid\mathbf x)
=
\frac{1}{24a^2}.
$$

The papers indicate that the correct potential hop should use the surface
Green-function PDF, not the uniform cube-surface PDF.

## Gaussian surface choice in code

For each observation net \(i\), the current implementation constructs one
axis-aligned Gaussian box around the merged net. If the observation net's
bounding box is

$$
[\mathbf b_{\min}, \mathbf b_{\max}],
$$

and the configured padding is \(d_G\), then the Gaussian box is

$$
[\mathbf b_{\min}-d_G\mathbf 1,\ \mathbf b_{\max}+d_G\mathbf 1].
$$

The box must lie strictly inside the FRW outer reference boundary. Sample points
are drawn uniformly by surface area over the six faces. For the correct
QuickCap/RWCap estimator, these sampled points are paired with the \(\omega\)
weight above.

## Outer reference boundary

The random walk must eventually terminate. For the current free-space prototype,
`FRWSolver` constructs an artificial outer reference box rather than using
`CapacitanceProblem.domain`. The problem domain remains a geometry-validity
constraint; the FRW outer box is the absorbing reference boundary.

Let the bounding box of all conductor geometry be

$$
[\mathbf g_{\min}, \mathbf g_{\max}],
$$

with center

$$
\mathbf g_c=\frac{\mathbf g_{\min}+\mathbf g_{\max}}{2},
$$

and largest geometry dimension

$$
L_g=\max_\alpha (g_{\max,\alpha}-g_{\min,\alpha}).
$$

For `outer_box_scale = s_o`, the outer reference boundary is the cube

$$
\left[
\mathbf g_c-\frac{s_oL_g}{2}\mathbf 1,\,
\mathbf g_c+\frac{s_oL_g}{2}\mathbf 1
\right].
$$

The default is

$$
s_o=20.
$$

If a walk reaches this outer boundary before hitting a conductor, it terminates
on the reference node at \(0\ \text{V}\). The Gaussian boxes must lie inside
this outer reference boundary.

## Matrix assembly

FRW estimates one observation row at a time in the Gaussian-surface flux
formulation. For observation conductor \(i\), each completed walk produces a
random vector \(\mathbf X_i\), and

$$
\widehat{\mathbf C}_{i,:}
=
\frac{1}{M_i}\sum_{m=1}^{M_i}\mathbf X_{i,m}.
$$

With the correct Green-function weight,

$$
\mathbf X_{i,m}
=
\omega(\mathbf r_m,\mathbf r_m^{(1)})\mathbf e_{h_m},
$$

where \(h_m\) is the conductor hit by the continuation walk. The reference
boundary contributes zero.

The public solver API still returns a matrix satisfying

$$
\mathbf Q=\mathbf C\mathbf V.
$$

As with BEM, the free-space reduced matrix may be augmented with an analytic
reference node so that

$$
\mathbf C^{\text{aug}}\mathbf 1=\mathbf 0.
$$

## Statistics and stopping

FRW is stochastic, so it needs explicit accuracy controls. The current solver
uses `samples_per_observation_net` as a maximum walk cap, not necessarily as the
actual number of walks. For each observation net, it runs at least
`min_samples_per_observation_net` walks, then checks the estimated error every
`check_interval` walks.

For a row entry \(C_{ik}\), the sample variance is

$$
s_{ik}^2
=
\frac{1}{M_i-1}
\sum_{m=1}^{M_i}
\left(X_{i,m}^{(k)}-\widehat C_{ik}\right)^2,
$$

and the standard error is

$$
\operatorname{SE}(\widehat C_{ik})
=
\frac{s_{ik}}{\sqrt{M_i}}.
$$

The relative standard error target is

$$
\frac{\operatorname{SE}(\widehat C_{ik})}{|\widehat C_{ik}|}<\tau,
$$

and the absolute standard error target is

$$
\operatorname{SE}(\widehat C_{ik})<\eta.
$$

For small coupling coefficients, relative error can be unstable, so the solver
supports both thresholds. In code, an entry is considered converged when

$$
\operatorname{SE}(\widehat C_{ik})
\le
\max\left(
  \eta,\,
  \tau|\widehat C_{ik}|
\right),
$$

using only the terms whose tolerances are configured. If neither tolerance is
set, the solver performs the full `samples_per_observation_net` walks. The
maximum cap is retained to prevent no-stop situations.

## Implementation checklist

Implemented now:

- `FRWSolver.solve(problem)` returns `FRWResult`;
- `FRWResult.capacitance` contains a Monte Carlo estimate;
- `FRWResult.standard_error` contains per-entry independent-sample standard
  errors for the reduced matrix;
- `FRWStatistics` records sample caps, actual walks per observation net, seed,
  transition parameters, completed walks, and escaped walks;
- `CenteredCubeGreenSampler` tabulates the centered-cube potential PDF;
- `CenteredCubeGreenSampler` evaluates the closed-form centered-cube
  \(-\nabla_{\mathbf r}G_\phi\) series for the first-step \(\omega\) weight;
- `create_solver("bem" | "frw")` selects between solver backends;
- examples visualize conductor geometry, Gaussian surfaces, representative
  walks, and transition cubes.

Still required before trusting FRW against BEM:

- validate the \(\omega\) estimator against BEM on simple two-box and
  one-box-to-reference examples;
- decide whether to keep sampling the first step from the potential PDF with
  an importance correction, or switch to a PDF proportional to
  \(|\nabla_{\mathbf r}P\cdot n|\) with the corresponding signed scale;
- add variance reduction and better space management;
- establish convergence criteria before using FRW labels for DNN training.

## Important papers and implementations

- `ref/00 BachelorThesis_FRW_and_Space_Management_NianlongGu.pdf`: thesis
  notes on FRW and space management. Useful for the Gaussian-surface charge
  integral and practical implementation details.

- `ref/06 A floating random-walk algorithm for extracting electrical capacitance.pdf`:
  the QuickCap theory paper. It describes the field/weight first step from a
  Gaussian surface and the subsequent potential random walk.

- `ref/08 RWCap_A_Floating_Random_Walk_Solver_for_3-D_Capacitance_Extraction_of_Very-Large-Scale_Integration_Interconnects.pdf`:
  the central RWCap paper. It states the \(\omega\) weight in terms of the
  gradient of the surface Green function and adds modern acceleration machinery.

- Ralph B. Iverson and Yannick L. Le Coz, "A floating random-walk algorithm for
  extracting electrical capacitance," Mathematics and Computers in Simulation,
  55(1-3), 59-66, 2001. DOI:
  [10.1016/S0378-4754(00)00246-9](https://doi.org/10.1016/S0378-4754(00)00246-9).

- Wenjian Yu, Hao Zhuang, Chao Zhang, Gang Hu, and Zhi Liu, "RWCap: A Floating
  Random Walk Solver for 3-D Capacitance Extraction of Very-Large-Scale
  Integration Interconnects," IEEE TCAD, 32(3), 353-366, 2013. DOI:
  [10.1109/TCAD.2012.2224346](https://doi.org/10.1109/TCAD.2012.2224346).
