import numpy as np

from geocouple.mesh.adapt import zz_estimator


def _cube_tets(n=4):
    """Structured tetrahedral mesh of the unit cube (Kuhn subdivision)."""
    g = np.linspace(0, 1, n + 1)
    idx = lambda i, j, k: (i * (n + 1) + j) * (n + 1) + k
    pts = np.array([(x, y, z) for x in g for y in g for z in g])
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                for perm in ([0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]):
                    tets.append([c[p] for p in perm])
    return pts, np.array(tets)


def test_zz_zero_for_linear_field():
    pts, tets = _cube_tets()
    v = pts @ np.array([1.0, -2.0, 0.5])
    eta2, vol, energy = zz_estimator(pts[tets], v[tets], tets, 1.0)
    assert vol.sum() == np.float64(vol.sum()) and abs(vol.sum() - 1) < 1e-12
    assert energy == np.float64(energy) and abs(energy - 5.25) < 1e-9
    assert eta2.max() < 1e-20


def test_zz_detects_curvature_and_decreases_with_refinement():
    errs = []
    for n in (4, 8):
        pts, tets = _cube_tets(n)
        v = (pts ** 2).sum(axis=1)
        eta2, _, energy = zz_estimator(pts[tets], v[tets], tets, 1.0)
        errs.append(np.sqrt(eta2.sum() / energy))
    assert errs[0] > 0 and errs[1] < errs[0]
