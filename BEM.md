# Dense BEM Capacitance Solver

This note documents the current low-order dense BEM method in `DenseBEMSolver`.
The implementation assumes a homogeneous dielectric with permittivity
\(\epsilon\) and axis-aligned cuboid conductors. Coordinates are continuous; no
integer-grid assumption is used by BEM.

## Net preprocessing

Input cuboids are first merged into electrical nets. Two cuboids are treated as
the same net when their closed-set intersection has dimension \(2\) or \(3\):

- \(3\)D overlap means their volumes overlap.
- \(2\)D overlap means they share a face patch with nonzero area.

Line contact and point contact are not merged.

After merging, only the exterior surface of each net is meshed. Internal faces
created by overlapping or face-touching cuboids are removed.

## Surface discretization

Each net surface is divided into rectangular panels. The current basis is
constant charge density per panel, and the collocation point is the panel
center.

For panel \(j\):

- center: \(\mathbf r_j\)
- area: \(A_j\)
- unknown constant surface charge density: \(\sigma_j\)
- net index: \(n(j)\)

The total charge on net \(i\) is approximated by summing the charges on all
panels belonging to that net:

$$
Q_i = \sum_{j:\,n(j)=i} \sigma_j A_j .
$$

## Influence matrix

The potential at collocation point \(\mathbf r_i\) due to panel \(j\) is first
approximated by treating the panel charge through the free-space Green's
function evaluated at the panel center:

$$
\phi_i \approx \sum_j M_{ij}\sigma_j ,
$$

with

$$
M_{ij}
= \frac{A_j}{4\pi\epsilon\,\lVert \mathbf r_i-\mathbf r_j\rVert},
\qquad i\ne j .
$$

This is the monopole term of the disk expansion. The implementation uses it for
far interactions. For nearer off-diagonal interactions, it replaces the point
center coefficient with the equal-area disk multipole expansion described
below.

This gives the dense linear system:

$$
\mathbf M\boldsymbol\sigma = \boldsymbol\phi ,
$$

where \(\phi_i\) is the prescribed conductor potential at the collocation
point.

## Self term: equal-area disk approximation

The off-diagonal formula is singular when \(i=j\), so the diagonal term uses an
equal-area disk approximation.

For panel \(i\) with area \(A_i\), define the radius of an equal-area disk:

$$
R_i = \sqrt{\frac{A_i}{\pi}} .
$$

For a uniformly charged disk with surface charge density \(\sigma\), the
potential at a target point \(\mathbf r\) is generally

$$
\phi(\mathbf r)
= \frac{\sigma}{4\pi\epsilon}
  \int_{\text{disk}} \frac{dA'}{\lVert \mathbf r-\mathbf r'\rVert}.
$$

If the disk lies in the \(xy\)-plane and the target point is represented in
spherical coordinates \((r,\theta,\varphi)\), symmetry removes the dependence on
\(\varphi\). With source coordinates \((\rho,\alpha)\) on the disk,

$$
\phi(r,\theta)
= \frac{\sigma}{4\pi\epsilon}
  \int_0^{R_i}\int_0^{2\pi}
  \frac{\rho\,d\alpha\,d\rho}
  {\sqrt{r^2+\rho^2-2r\rho\sin\theta\cos\alpha}} .
$$

This integral is the general axisymmetric external potential of the disk. It
does not reduce to a simple elementary expression for an arbitrary off-axis
point. For \(r>R_i\), it can be written as the convergent multipole expansion

$$
\phi(r,\theta)
= \frac{\sigma}{2\epsilon}
  \sum_{\ell=0}^{\infty}
  \frac{R_i^{\ell+2}}{(\ell+2)r^{\ell+1}}
  P_\ell(0)P_\ell(\cos\theta),
\qquad r>R_i ,
$$

where \(P_\ell\) is the Legendre polynomial. Odd \(\ell\) terms vanish because
\(P_\ell(0)=0\).

For an off-diagonal source panel \(j\), define the equal-area disk radius

$$
R_j = \sqrt{\frac{A_j}{\pi}},
$$

and let \(r=\lVert \mathbf r_i-\mathbf r_j\rVert\). The multipole expansion is
used only in its convergence domain \(r>R_j\). For \(r>kR_j\), the code switches
back to the point-center approximation.

If an off-diagonal interaction has \(r\le R_j\), the multipole series is not
used because it is outside its convergence domain. The current prototype falls
back to the point-center coefficient for that case. For higher accuracy, this
near-field corner should be replaced by rectangular-panel analytic integration
or a dedicated numerical quadrature.

The default is

$$
k=5.
$$

This value comes from scanning \(\theta\in[0,\pi]\) and finding where the
worst-case relative difference between the full disk multipole and the
point-center monopole falls below \(1\%\):

$$
\max_\theta
\frac{
  \left|\phi_{\text{point}}(r)-\phi_{\text{disk}}(r,\theta)\right|
}{
  \left|\phi_{\text{disk}}(r,\theta)\right|
}
< 0.01,
\qquad r>kR .
$$

Numerically, the threshold is very close to \(k=5\), with the worst case on the
disk axis.

For the diagonal term we only need the potential at the center of the disk on
its surface. There the distance from the target to a source point is \(\rho\),
and \(dA'=\rho\,d\rho\,d\alpha\), so

$$
\begin{aligned}
\phi_{\text{center}}
&= \frac{\sigma}{4\pi\epsilon}
   \int_0^{2\pi}\int_0^{R_i}
   \frac{\rho\,d\rho\,d\alpha}{\rho} \\
&= \frac{\sigma}{4\pi\epsilon}
   \int_0^{2\pi}\int_0^{R_i}
   d\rho\,d\alpha \\
&= \frac{\sigma}{4\pi\epsilon}(2\pi R_i) \\
&= \frac{\sigma R_i}{2\epsilon}.
\end{aligned}
$$

Therefore the diagonal influence entry is:

$$
M_{ii}
= \frac{R_i}{2\epsilon}
= \frac{1}{2\epsilon}\sqrt{\frac{A_i}{\pi}} .
$$

This is a first-order approximation. It is simple and stable for the prototype,
but can later be replaced with an analytic rectangular-panel self integral.

## Building the capacitance matrix

Let there be \(N\) merged conductor nets and \(P\) surface panels.

Define a panel-to-net potential matrix \(\mathbf V_{\text{panel}}\in
\mathbb R^{P\times N}\):

$$
V_{\text{panel},jk}
=
\begin{cases}
1, & n(j)=k,\\
0, & n(j)\ne k.
\end{cases}
$$

Column \(k\) corresponds to the electrostatic experiment:

$$
V_k = 1\ \text{V}, \qquad V_m = 0\ \text{V}\quad(m\ne k).
$$

The panel charge densities for all \(N\) right-hand sides satisfy:

$$
\mathbf M\boldsymbol\Sigma = \mathbf V_{\text{panel}},
$$

where

$$
\Sigma_{jk}
= \text{charge density on panel }j
  \text{ when net }k\text{ is driven to }1\ \text{V}.
$$

For each driven net \(k\), the charge on observation net \(i\) is:

$$
C_{ik}
= Q_i^{(k)}
= \sum_{j:\,n(j)=i} \Sigma_{jk}A_j .
$$

Thus the capacitance matrix \(\mathbf C\) is built by summing panel charge over
each observation net for every excitation column.

The current Python implementation loops over the \(N\) right-hand sides:

```text
for driven_net k:
    solve M sigma^(k) = V_panel[:, k]
    C[i, k] = sum_{j: n(j)=i} sigma_j^(k) A_j
```

Mathematically this is equivalent to solving the multi-RHS system

$$
\mathbf M\boldsymbol\Sigma = \mathbf V_{\text{panel}}
$$

once. A future optimization can reuse a single matrix factorization for all
\(N\) right-hand sides.

## Symmetry

The continuous Maxwell capacitance matrix is symmetric by reciprocity. Because
the current method uses a low-order collocation discretization and approximate
self terms, small asymmetries can appear numerically. The solver therefore
symmetrizes the result by default:

$$
\mathbf C \leftarrow \frac{1}{2}(\mathbf C+\mathbf C^\mathsf T).
$$

## Reference node / enclosure augmentation

The reduced matrix from BEM treats the omitted reference conductor as held at
\(0\ \text{V}\). In the current free-space interpretation, that reference is
infinity. When `add_reference_node=True`, the solver appends an explicit
reference net named `enclosure` by default.

Given an \(N\times N\) reduced matrix \(\mathbf C\), the augmented
\((N+1)\times(N+1)\) matrix is:

$$
C^{\text{aug}}_{ij}=C_{ij},
\qquad 0\le i,j<N,
$$

$$
C^{\text{aug}}_{iN}
= -\sum_{j=0}^{N-1} C_{ij},
\qquad 0\le i<N,
$$

$$
C^{\text{aug}}_{Nj}
= -\sum_{i=0}^{N-1} C_{ij},
\qquad 0\le j<N,
$$

and

$$
C^{\text{aug}}_{NN}
= \sum_{i=0}^{N-1}\sum_{j=0}^{N-1} C_{ij}.
$$

This makes every row and column sum to zero:

$$
\mathbf C^{\text{aug}}\mathbf 1 = \mathbf 0 .
$$

So if all conductors, including the reference net, are floating at the same
potential, every net has zero net charge. For the complete Maxwell matrix the
expected sign pattern is positive diagonal and non-positive off-diagonal terms.

This augmentation enforces the circuit-level reference-node property. It does
not yet model the geometric effect of a finite metal enclosure at a specific
distance; that would require including a real enclosing boundary in the BEM
geometry.
