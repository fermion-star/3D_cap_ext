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

For the first `capext` FRW implementation, the planned order is:

1. single-dielectric axis-aligned boxes;
2. cube or sphere transition domain selected from distance to nearest conductor;
3. absorbing conductor hit detection;
4. simple independent-walk statistics;
5. later: importance sampling, stratified sampling, octree space management.

## Matrix assembly

FRW can estimate one row or one column at a time depending on the chosen
formulation. For our software interface, the solver should still return a
matrix satisfying

$$
\mathbf Q = \mathbf C\mathbf V.
$$

The initial framework will keep the same public API as BEM:

```python
solver.solve_matrix(problem)
```

and later return:

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

FRW is stochastic, so it needs explicit accuracy controls. Candidate stopping
criteria:

- fixed walks per observation net;
- fixed walks per matrix entry;
- relative standard error target:

$$
\frac{\operatorname{SE}(\widehat C_{ik})}{|\widehat C_{ik}|}
< \tau;
$$

- absolute standard error target:

$$
\operatorname{SE}(\widehat C_{ik}) < \eta.
$$

For small coupling coefficients, relative error can be unstable, so the solver
should support both relative and absolute stopping thresholds.

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

## First code milestone

The first code milestone is intentionally a framework, not a claimed numerical
FRW solver:

- add `FRWSolver` with the same `solve_matrix(problem)` interface as BEM;
- add stochastic configuration: samples, seed, tolerances;
- add result/statistics dataclasses;
- add `create_solver("bem" | "frw")` factory;
- keep `FRWSolver.solve_matrix` raising `NotImplementedError` until the walk
  kernel is implemented.

This keeps the public API stable while preventing accidental use of an
unfinished stochastic solver as if it were producing valid capacitance numbers.
