examples/run_bem.py
===================

Geometry:

[
    ("left", (100.0, 220.0, 220.0), (160.0, 280.0, 280.0)),
    ("right", (260.0, 220.0, 220.0), (320.0, 280.0, 280.0)),
    ("left_extension", (160.0, 230.0, 230.0), (190.0, 270.0, 270.0)),
]

Merged nets:

('left+left_extension', 'right')

# BEM, disk approximation
## study 1: symmetrize
BEM unknown counts:

without physical outer box: 150
with meshed physical outer box: 3900

Current solve:

The reference-node enclosure is appended analytically, so it does not add BEM
unknowns. The dense influence matrix solved for this example is 150 x 150 with
2 RHS columns.

symmetrize = true

[[ 5.102710e-09 -1.353685e-09 -3.749025e-09]
 [-1.353685e-09  4.703415e-09 -3.349730e-09]
 [-3.749025e-09 -3.349730e-09  7.098755e-09]]

symmetrize = false

[[ 5.102710388083e-09 -1.353586978564e-09 -3.749123409519e-09]
 [-1.353783198873e-09  4.703415143085e-09 -3.349631944212e-09]
 [-3.748927189210e-09 -3.349828164520e-09  7.098755353731e-09]]

symmetrize = false row sums:

[0. 0. 0.]

symmetrize = false asymmetry C - C.T:

[[ 0.000000000000e+00  1.962203082399e-13 -1.962203082399e-13]
 [-1.962203082399e-13  0.000000000000e+00  1.962203082399e-13]
 [ 1.962203082399e-13 -1.962203082399e-13  0.000000000000e+00]]

## study 2: mesh refinement
influence matrix size: (150, 150)
max_panel_size=20, unknowns=150
[[ 5.102710388083e-09 -1.353685088719e-09 -3.749025299365e-09]
 [-1.353685088719e-09  4.703415143085e-09 -3.349730054366e-09]
 [-3.749025299365e-09 -3.349730054366e-09  7.098755353731e-09]]

influence matrix size: (250, 250)
max_panel_size=15, unknowns=250
[[ 5.140797423329e-09 -1.374323465636e-09 -3.766473957693e-09]
 [-1.374323465636e-09  4.745488420032e-09 -3.371164954397e-09]
 [-3.766473957693e-09 -3.371164954397e-09  7.137638912090e-09]]

influence matrix size: (480, 480)
max_panel_size=10, unknowns=480
[[ 5.166103117203e-09 -1.392618186215e-09 -3.773484930988e-09]
 [-1.392618186215e-09  4.785322833179e-09 -3.392704646964e-09]
 [-3.773484930988e-09 -3.392704646964e-09  7.166189577952e-09]]

influence matrix size: (1920, 1920)
max_panel_size=5, unknowns=1920
[[ 5.205767942434e-09 -1.412947865457e-09 -3.792820076977e-09]
 [-1.412947865457e-09  4.821478931099e-09 -3.408531065642e-09]
 [-3.792820076977e-09 -3.408531065642e-09  7.201351142619e-09]]

relative difference vs max_panel_size=5 (Frobenius norm)
20: 1.827916e-02
15: 1.158122e-02
10: 6.256171e-03
