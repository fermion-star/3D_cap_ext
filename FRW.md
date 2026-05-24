# Floating Random Walk Capacitance Solver

This note records the intended Floating Random Walk (FRW) solver design for
`capext`. The first target is a homogeneous dielectric with axis-aligned cuboid
conductors, matching the current BEM problem model.

## Physical problem

For a homogeneous dielectric, the potential in the dielectric region satisfies

$$
\nabla^2 \phi = 0.
$$

Each conductor net is an equipotential Dirichlet boundary. For the \(k\)-th
capacitance-column experiment,

$$
\phi = 1 \quad \text{on conductor } k,
$$

and

$$
\phi = 0 \quad \text{on all other conductors and the reference boundary}.
$$

The Maxwell capacitance coefficient is

$$
C_{ik}=Q_i^{(k)}
=-\int_{\Gamma_i}\epsilon\frac{\partial \phi^{(k)}}{\partial n}\,dS,
$$

where \(\Gamma_i\) is the surface of observation conductor \(i\), and \(n\) is
the outward normal from the dielectric region.

## FRW interpretation

FRW rewrites the boundary integral above as a Monte Carlo expectation. A single
sample is a random walk that starts near an observation conductor surface and
jumps through a sequence of local transition domains until it hits a conductor
boundary.

For a given observation conductor \(i\), a sample contributes a random variable

$$
X_{i}^{(k)}
$$

to the charge estimate under excitation \(k\). In the simplest hitting
probability view, this contribution is proportional to an indicator:

$$
\mathbf 1\{\text{walk terminates on conductor }k\}.
$$

Practical FRW capacitance solvers use a weighted formulation. The weight
accounts for the starting surface distribution, Green's function normal
derivative, and transition-domain exit probabilities. In abstract form:

$$
Q_i^{(k)}
= \mathbb E[X_i^{(k)}].
$$

With \(M\) independent walks, the estimator is

$$
\widehat C_{ik}
= \widehat Q_i^{(k)}
= \frac{1}{M}\sum_{m=1}^M X_{i,m}^{(k)}.
$$

The sample variance is

$$
s_{ik}^2
= \frac{1}{M-1}
   \sum_{m=1}^M
   \left(X_{i,m}^{(k)}-\widehat C_{ik}\right)^2,
$$

and the standard error is

$$
\operatorname{SE}(\widehat C_{ik})
= \frac{s_{ik}}{\sqrt M}.
$$

A normal-approximation confidence interval is

$$
\widehat C_{ik}
\pm
z_{1-\alpha/2}\operatorname{SE}(\widehat C_{ik}).
$$

For a 95% interval, \(z_{1-\alpha/2}\approx 1.96\).

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

The first `capext` FRW implementation uses:

1. single-dielectric axis-aligned boxes;
2. the largest cubic transition domain centered at the current random-walk
   point;
3. a discretized centered-cube surface Green-function table for the exit PDF;
4. absorbing conductor and outer-reference-boundary hit detection;
5. independent-walk statistics with optional error-based stopping.

This is intentionally still a prototype. The hop PDF now follows the
single-dielectric centered-cube Green-function series used by QuickCap-style
FRW, but the Gaussian-surface capacitance weight is still a low-order
finite-distance approximation rather than the full weight-value formulation.

## Transition cube Green-function PDF currently used in code

At random-walk point \(\mathbf x\), define the cubic transition domain

$$
D(\mathbf x,a)
=
\left\{
\mathbf y:\ \lVert \mathbf y-\mathbf x\rVert_\infty < a
\right\},
$$

where the half-size is

$$
a = s\,d_\infty(\mathbf x).
$$

Here \(s=\texttt{transition\_safety}\), and \(d_\infty(\mathbf x)\) is the
minimum \(L_\infty\) distance from \(\mathbf x\) to any conductor or the FRW
outer boundary. By default,

$$
s=1,
$$

so the code uses the maximum conductor-free transition cube, matching the
QuickCap/RWCap description. Smaller values of \(s\) can be used only as a
debugging/robustness knob.

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
on the reference node at \(0\ \text{V}\).

The Gaussian boxes must lie inside this outer reference boundary.

The current `frw.py` implementation uses the single-dielectric surface Green
function for a cube centered at the current random-walk point. This follows the
basic FRW relation

For a point \(\mathbf r\) inside a closed transition surface \(S\), RWCap writes
the potential as

$$
\phi(\mathbf r)
=
\oint_S P(\mathbf r,\mathbf r^{(1)})
\phi(\mathbf r^{(1)})\,dS_{\mathbf r^{(1)}} ,
$$

where \(P(\mathbf r,\mathbf r^{(1)})\) is the surface Green function. For fixed
\(\mathbf r\), it is a PDF on \(S\):

$$
\oint_S P(\mathbf r,\mathbf r^{(1)})\,dS_{\mathbf r^{(1)}}
=1.
$$

For the homogeneous cube, this PDF can be derived analytically and tabulated.
Using a normalized unit cube \([0,1]^3\), source point at the center
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

This is very different from the earlier uniform approximation

$$
p_{\text{uniform}}(\mathbf y\mid\mathbf x)
=
\frac{1}{24a^2},
$$

which would choose a face uniformly and then sample uniformly on that face. The
papers indicate that the correct FRW hop should use the surface Green-function
PDF, not the uniform cube-surface PDF.

The remaining limitation is not the hop PDF for single-dielectric centered
cubes; it is the capacitance weight used on the first cube from the Gaussian
surface. RWCap uses a weight value involving the gradient of the surface Green
function:

$$
\omega(\mathbf r,\mathbf r^{(1)})
=
-\frac{
\nabla_{\mathbf r}P(\mathbf r,\mathbf r^{(1)})\cdot\hat{\mathbf n}(\mathbf r)
}{
g\,P(\mathbf r,\mathbf r^{(1)})
}.
$$

The current code still uses a simpler finite-distance Gaussian-box flux weight.

## Gaussian surface choice

For each observation net \(i\), the current implementation constructs one
axis-aligned Gaussian box around the merged net. If the observation net's
bounding box is

$$
[\mathbf b_{\min}, \mathbf b_{\max}],
$$

and the configured padding is \(g\), then the Gaussian box is

$$
[\mathbf b_{\min}-g\mathbf 1,\ \mathbf b_{\max}+g\mathbf 1].
$$

The box must lie strictly inside the FRW outer reference boundary. Sample points
are drawn uniformly by surface area over the six faces.

The present charge estimator uses a first-order finite-distance flux model:

$$
Q_i^{(k)}
\approx
\frac{\epsilon A_G}{g}
\mathbb E\left[V_i^{(k)}-\phi^{(k)}(\mathbf X_G)\right],
$$

where \(A_G\) is the Gaussian-box surface area and \(\mathbf X_G\) is a uniform
random point on that surface. Under excitation \(k\),

$$
V_i^{(k)}=
\begin{cases}
1, & i=k,\\
0, & i\ne k.
\end{cases}
$$

The potential sample is estimated by the conductor hit identity:

$$
\phi^{(k)}(\mathbf X_G)
\approx
\mathbf 1\{\text{walk from }\mathbf X_G\text{ hits conductor }k\}.
$$

Therefore a single walk starting from observation net \(i\)'s Gaussian surface
contributes the vector

$$
\mathbf X_i
=
\frac{\epsilon A_G}{g}
\left(\mathbf e_i-\mathbf e_h\right),
$$

where \(h\) is the hit conductor index. If the walk exits through the FRW outer
reference boundary, \(\mathbf e_h\) is omitted, corresponding to the reference
boundary at \(0\ \text{V}\).

This estimator is useful for building the solver structure and statistics, but
the Gaussian-box finite-difference weight is still a low-order approximation.
Later versions should replace it with the standard FRW Green-function normal
derivative weight for the chosen Gaussian surface.

## Matrix assembly

FRW can estimate one row or one column at a time depending on the chosen
formulation. For our software interface, the solver should still return a
matrix satisfying

$$
\mathbf Q = \mathbf C\mathbf V.
$$

The FRW solver keeps the same public API as BEM:

```python
solver.solve_matrix(problem)
```

and returns:

- capacitance estimate \(\widehat{\mathbf C}\);
- per-entry standard errors;
- number of walks;
- random seed and stopping criteria.

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

The relative standard error target is:

$$
\frac{\operatorname{SE}(\widehat C_{ik})}{|\widehat C_{ik}|}
< \tau;
$$

The absolute standard error target is:

$$
\operatorname{SE}(\widehat C_{ik}) < \eta.
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

## Important papers and implementations

- Ralph B. Iverson and Yannick L. Le Coz, "A floating random-walk algorithm for
  extracting electrical capacitance," Mathematics and Computers in Simulation,
  55(1-3), 59-66, 2001. This is the QuickCap theory paper and describes FRW as
  Monte Carlo integration where one sample corresponds to one floating random
  walk. DOI:
  [10.1016/S0378-4754(00)00246-9](https://doi.org/10.1016/S0378-4754(00)00246-9).

- Yannick L. Le Coz and Ralph B. Iverson, "A stochastic algorithm for high speed
  capacitance extraction in integrated circuits," Solid-State Electronics,
  1992. This is an early QuickCap-era FRW paper. Publisher page:
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0038110192903327).

- Yannick L. Le Coz, Ralph B. Iverson, et al., "Performance of random-walk
  capacitance extractors for IC interconnects: A numerical study,"
  Solid-State Electronics, 1998. This paper compares random-walk capacitance
  extraction against analytical, deterministic, and experimental references.
  Publisher page:
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0038110197002839).

- Wenjian Yu, Hao Zhuang, Chao Zhang, Gang Hu, and Zhi Liu, "RWCap: A Floating
  Random Walk Solver for 3-D Capacitance Extraction of Very-Large-Scale
  Integration Interconnects," IEEE TCAD, 32(3), 353-366, 2013. This is the
  central RWCap paper; it includes multi-dielectric transition probability /
  weight characterization, variance reduction, octree space management, and
  parallelism. DOI:
  [10.1109/TCAD.2012.2224346](https://doi.org/10.1109/TCAD.2012.2224346).
  Open PDF:
  [Tsinghua NUMBDA](https://numbda.cs.tsinghua.edu.cn/papers/tcad13.pdf).

- Wenjian Yu and Xiren Wang, "Fast Floating Random Walk Method for Capacitance
  Extraction," in Advanced Field-Solver Techniques for RC Extraction of
  Integrated Circuits, Springer, 2014. This book chapter is a useful survey of
  FRW techniques and acceleration ideas. DOI:
  [10.1007/978-3-642-54298-5_10](https://doi.org/10.1007/978-3-642-54298-5_10).

- RWCap project pages from Wenjian Yu's group provide practical solver context
  and lists of related publications:
  [RWCap v1/v2](https://numbda.cs.tsinghua.edu.cn/download/RWCap_v1_en.html),
  [RWCap v4](https://numbda.cs.tsinghua.edu.cn/download/RWCap_v4_en.html).

- Ming Yang and Wenjian Yu, "Floating Random Walk Capacitance Solver Tackling
  Conformal Dielectric with On-the-Fly Sampling on Eight-Octant Transition
  Cubes," IEEE TCAD, 39(12), 4935-4943, 2020. This is important for later
  multi/conformal dielectric support. DOI:
  [10.1109/TCAD.2020.2968544](https://doi.org/10.1109/TCAD.2020.2968544).

## Current code milestone

The current code milestone is a runnable prototype:

- `FRWSolver.solve(problem)` returns `FRWResult`;
- `FRWResult.capacitance` contains a Monte Carlo estimate;
- `FRWResult.standard_error` contains per-entry independent-sample standard
  errors for the reduced matrix;
- `FRWStatistics` records max sample count, actual walks per observation net,
  seed, transition parameters, completed walks, and escaped walks;
- `create_solver("bem" | "frw")` selects between solver backends.

The prototype is suitable for exercising the API, studying random-walk
statistics, and comparing qualitative trends. It should not yet be used as a
trusted replacement for BEM labels.
