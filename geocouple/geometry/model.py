"""Asamblarea unui model FEM: primitive + cub de aer + strat de transformare la infinit, grupuri fizice, mesh.

Convenția de grupuri fizice (aceleași numere ajung ca Region[] în GetDP, deci nu se suprapun între dimensiuni):

    volum primitivă          = id-ul nodului din graf            (1 … 999)
    aer interior (cub)       = 1001
    strat de transformare    = 1011, 1012, 1013   (prismele învelișului cubic, după axa feței: x, y, z)
    frontiera exterioară     = 1003   (suprafață; „infinitul” când există strat de transformare)
    port                     = 10000 + 10 · id_nod + indice_port (1-based, în ordinea porturilor primitivei)
    pielea unui conductor    = 20000 + id_conductor            (suprafață, fără fețele interne dintre primitive)
    contact între primitive  = 30000 + indice_contact (1-based) (fețele comune după fragmentare)

Fiecare tetraedru primește un nod: cele din conductoare nodul primitivei căreia îi aparțin (grupul fizic), cele din aer
cea mai apropiată primitivă (`element_nodes.npz`). Prismele învelișului nu primesc nod.

Domeniul exterior este un cub de aer de semilatură a, înconjurat de un înveliș cubic [a, b] care se aplică pe tot
spațiul exterior (b ↦ infinit), cu aplicația lui VolRectShell din GetDP scrisă ca material anizotrop echivalent
(physics/templates/common.pro). Ponderea tangențială a transformării diverge spre b, așa că învelișul se discretizează
structurat: straturi de prisme obținute prin scalarea suprafeței cubului din centru (`_add_shell_layers`).
Variante încercate și respinse în P0 (eroare pe capacitate): VolSphShell pe tetraedre (+10–20%), VolSphShell pe prisme
(fațetele plane nu stau pe sfere, +14%), VolRectShell din GetDP 4.0 pe tetraedre sau prisme (+50…200%).

Geometria se construiește în µm; mesh-ul se scrie în metri (Mesh.ScalingFactor = 1e-6).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from geocouple.geometry.primitives import Primitive
from geocouple.stack import Stack

UM = 1e-6
AIR_TAG, INF_TAG = 1001, 1003
SHELL_TAGS = {1: 1011, 2: 1012, 3: 1013}  # transformation shell, by face axis


class MeshQualityError(RuntimeError):
    pass


def port_tag(node_id: int, port_index: int) -> int:
    return 10000 + 10 * node_id + port_index


def skin_tag(conductor_id: int) -> int:
    return 20000 + conductor_id


def contact_tag(index: int) -> int:
    return 30000 + index


@dataclass
class Conductor:
    """An electrical conductor: a set of primitives plus the two external ports used for DC excitation."""

    id: int
    nodes: list[int]
    port_hi: tuple[int, str] | None  # (node id, port name); None: no DC excitation (e.g. imported polygon only)
    port_lo: tuple[int, str] | None


@dataclass
class DomainSpec:
    r_in_factor: float = 1.5     # air cube half-size = factor × largest half-extent of the geometry bbox
    r_out_ratio: float = 1.25    # shell outer half-size = ratio × inner half-size
    with_shell: bool = True      # False: Dirichlet directly on the air cube (no transformation)


@dataclass
class MeshSpec:
    cells_across: float = 4.0       # elements across the smallest conductor cross-section dimension
    growth: float = 0.15            # element size ≈ growth × distance from the nearest conductor
    shell_layers: int = 4           # prism layers across the transformation shell
    port_refine: float = 1.0        # size factor at ports and corners (< 1 refines them; graph-based coarse sizing)
    optimize: bool = True           # gmsh + Netgen optimization after meshing
    algorithm3d: int = 1            # 1 = Delaunay (robust); 10 = HXT (faster, but produced near-zero-volume tets)
    min_quality: float = 0.0        # reject meshes whose minimum SICN is below this
    max_tets: int = 1_000_000


@dataclass
class ModelInfo:
    msh: str
    center_m: list[float]
    r_in_m: float                 # half-size of the air cube
    r_out_m: float | None         # half-size of the shell cube (None: no transformation)
    n_tets: int
    n_prisms: int                 # transformation shell (3 tetrahedra each when counting against the cap)
    n_nodes: int
    mesh_time_s: float
    volumes: dict[int, float]                          # node id -> sigma [S/m]
    conductors: list[dict] = field(default_factory=list)
    eps_r_background: float = 1.0
    contacts: list[dict] = field(default_factory=list)  # {src, dst, group, area_um2}
    element_nodes: str | None = None                    # npz: tet tags, owner node (0 = air), nearest node
    min_quality: float | None = None                    # minimum SICN over all volume elements

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> ModelInfo:
        d = json.loads(Path(path).read_text())
        d["volumes"] = {int(k): v for k, v in d["volumes"].items()}
        return cls(**d)


def _surface_normal(tag: int) -> np.ndarray:
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    uv = [(a + b) / 2 for a, b in zip(lo, hi)]
    return np.asarray(gmsh.model.getNormal(tag, uv))


def _find_port_surfaces(candidates: list[int], center, normal, half_size: float, tol: float) -> list[int]:
    """Faces of the port: the face centred on the port, or — when fragmentation split the port face because other
    primitives abut it — every coplanar face of the primitive within the port's extent."""
    c, n = np.asarray(center), np.asarray(normal)
    exact, split = [], []
    for s in candidates:
        if abs(abs(_surface_normal(s) @ n) - 1) > 1e-6:
            continue
        com = np.asarray(gmsh.model.occ.getCenterOfMass(2, s))
        if abs((com - c) @ n) > tol:
            continue
        if np.linalg.norm(com - c) < tol:
            exact.append(s)
        elif np.linalg.norm(com - c) < half_size:
            split.append(s)
    if exact:
        return exact
    if not split:
        raise RuntimeError(f"port at {center} normal {normal}: no face found")
    return split


def _add_shell_layers(box_faces: list[int], center: np.ndarray, a: float, b: float, n_layers: int) -> int:
    """Extrude the meshed surface of the air cube into `n_layers` prism layers by scaling from the centre.

    Layer k is the inner surface mesh scaled by (a + (b - a) k / n) / a, so every level set of the VolRectShell
    "radius" (distance along the face axis) is a plane and every extrusion line passes through the centre. Inside each
    prism the mapped field (B - R) g(direction) is then interpolated without the error that unstructured tetrahedra
    make near the outer boundary, where the transformation weight diverges. Prisms go to the shell group of the axis
    normal to their base face; the outer surface becomes INF_TAG.
    """
    tris, node_tags, node_xyz = [], [], []
    for s in box_faces:
        types, _, elem_nodes = gmsh.model.mesh.getElements(2, s)
        if list(types) != [2]:
            raise RuntimeError("the air cube surface must be meshed with linear triangles")
        tris.append(np.asarray(elem_nodes[0], dtype=np.int64).reshape(-1, 3))
        tags, xyz, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
        node_tags.append(tags)
        node_xyz.append(np.reshape(xyz, (-1, 3)))
    tris = np.vstack(tris)
    tags, first = np.unique(np.concatenate(node_tags), return_index=True)
    rel = np.vstack(node_xyz)[first] - center
    local = np.searchsorted(tags, tris)

    # orient each base triangle outwards so that its prism has positive volume
    p0, p1, p2 = (rel[local[:, i]] for i in range(3))
    centroid = (p0 + p1 + p2) / 3
    flip = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), centroid) < 0
    local[flip] = local[flip][:, [0, 2, 1]]
    axis = np.argmax(np.abs(centroid), axis=1)

    volumes = {ax: gmsh.model.addDiscreteEntity(3) for ax in range(3)}
    outer = gmsh.model.addDiscreteEntity(2)
    layer_tags = [tags]
    next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    for k in range(1, n_layers + 1):
        new = np.arange(next_tag, next_tag + len(tags))
        next_tag += len(tags)
        scale = (a + (b - a) * k / n_layers) / a
        last = k == n_layers
        gmsh.model.mesh.addNodes(2 if last else 3, outer if last else volumes[0], new, (center + scale * rel).ravel())
        layer_tags.append(new)
    n_prisms = 0
    for ax, vol in volumes.items():
        sel = local[axis == ax]
        prisms = np.vstack([np.hstack([layer_tags[k][sel], layer_tags[k + 1][sel]]) for k in range(n_layers)])
        gmsh.model.mesh.addElementsByType(vol, 6, [], prisms.ravel())
        gmsh.model.addPhysicalGroup(3, [vol], SHELL_TAGS[ax + 1], name=f"shell_{'xyz'[ax]}")
        n_prisms += len(prisms)
    gmsh.model.mesh.addElementsByType(outer, 2, [], layer_tags[-1][local].ravel())
    gmsh.model.addPhysicalGroup(2, [outer], INF_TAG, name="infinity")
    return n_prisms


def _qualities() -> tuple[int, float]:
    tags = gmsh.model.mesh.getElementsByType(4)[0]
    q = gmsh.model.mesh.getElementQualities(tags, "minSICN")
    return len(tags), float(np.min(q))


def _element_nodes(primitives: list[Primitive], prim_vols: dict[int, list[int]], air: list[int], stack: Stack,
                   path: Path, spacing: float) -> Path:
    """Owner node of every tetrahedron (conductor tets: their primitive; air tets: 0) plus the nearest primitive."""
    from scipy.spatial import cKDTree

    pts, ids = [], []
    for p in primitives:
        sp = p.surface_points(stack, spacing)
        pts.append(sp)
        ids.append(np.full(len(sp), p.id))
    tree, ids = cKDTree(np.vstack(pts)), np.concatenate(ids)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    index[node_tags.astype(np.int64)] = np.arange(len(node_tags))
    coords = coords.reshape(-1, 3)
    all_tags, owners, nearest = [], [], []
    for owner, vols in [(p.id, prim_vols[p.id]) for p in primitives] + [(0, air)]:
        for v in vols:
            tags, nodes = gmsh.model.mesh.getElementsByType(4, v)
            if len(tags) == 0:
                continue
            cent = coords[index[nodes.astype(np.int64)]].reshape(-1, 4, 3).mean(axis=1)
            all_tags.append(tags)
            owners.append(np.full(len(tags), owner))
            nearest.append(np.full(len(tags), owner) if owner else ids[tree.query(cent)[1]])
    np.savez_compressed(path, tet=np.concatenate(all_tags), owner=np.concatenate(owners),
                        nearest=np.concatenate(nearest))
    return path


def build_model(primitives: list[Primitive], conductors: list[Conductor], stack: Stack, out_dir: Path,
                domain: DomainSpec = DomainSpec(), mesh: MeshSpec = MeshSpec(), name: str = "model",
                background: tuple[np.ndarray, np.ndarray] | None = None) -> ModelInfo:
    """background: optional (tets (N, 4, 3) in µm, nodal sizes (N, 4) in µm) — e.g. from the ZZ estimator; the mesh
    uses the minimum of this size and the graph-based size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [p.id for p in primitives]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate primitive ids")
    by_id = {p.id: p for p in primitives}
    for c in conductors:
        if not set(c.nodes) <= set(ids):
            raise ValueError(f"conductor {c.id} references unknown nodes")

    if domain.r_in_factor <= 1.0 or domain.r_out_ratio <= 1.0:
        raise ValueError("r_in_factor and r_out_ratio must be > 1")

    t0 = time.perf_counter()
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        occ = gmsh.model.occ

        inputs = [p.build(occ, stack) for p in primitives]
        boxes = [p.bbox(stack) for p in primitives]
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        center = (lo + hi) / 2
        r_in = domain.r_in_factor * float(np.max(hi - lo)) / 2
        r_out = domain.r_out_ratio * r_in if domain.with_shell else None
        flat = [dt for group in inputs for dt in group]
        air_box = (3, occ.addBox(*(center - r_in), 2 * r_in, 2 * r_in, 2 * r_in))
        _, out_map = occ.fragment(flat, [air_box])
        children = iter(out_map)
        prim_vols: dict[int, list[int]] = {}
        for p, group in zip(primitives, inputs):
            prim_vols[p.id] = sorted({t for _ in group for d, t in next(children) if d == 3})
        all_cond = [t for v in prim_vols.values() for t in v]
        if len(all_cond) != len(set(all_cond)):
            raise RuntimeError("overlapping primitives (shared volumes) are not supported: primitives may only touch")
        inner = {t for d, t in next(children) if d == 3}
        air = sorted(inner - set(all_cond))
        occ.synchronize()

        for p in primitives:
            p.volumes = prim_vols[p.id]
            gmsh.model.addPhysicalGroup(3, p.volumes, p.id, name=f"node_{p.id}_{p.type}")
        gmsh.model.addPhysicalGroup(3, air, AIR_TAG, name="air")

        # contacts: faces shared by two primitives after fragmentation
        faces_of = {p.id: {t for _, t in gmsh.model.getBoundary([(3, v) for v in p.volumes], combined=False,
                                                                oriented=False)} for p in primitives}
        contacts = []
        for i, pi in enumerate(primitives):
            for pj in primitives[i + 1:]:
                shared = sorted(faces_of[pi.id] & faces_of[pj.id])
                if shared:
                    tag = contact_tag(len(contacts) + 1)
                    gmsh.model.addPhysicalGroup(2, shared, tag, name=f"contact_{pi.id}_{pj.id}")
                    contacts.append({"src": pi.id, "dst": pj.id, "group": tag,
                                     "area_um2": float(sum(occ.getMass(2, f) for f in shared))})

        box_faces = sorted({t for _, t in gmsh.model.getBoundary([(3, v) for v in air], combined=True, oriented=False)
                            if np.isclose(np.abs(np.asarray(gmsh.model.occ.getCenterOfMass(2, t)) - center).max(),
                                          r_in, rtol=1e-6)})
        if len(box_faces) != 6:
            raise RuntimeError(f"expected 6 air cube faces, found {len(box_faces)}")
        if not r_out:
            gmsh.model.addPhysicalGroup(2, box_faces, INF_TAG, name="outer")

        feature = min(p.min_feature_um(stack) for p in primitives)
        conductor_info = []
        port_faces = []
        for p in primitives:
            faces = [t for _, t in gmsh.model.getBoundary([(3, v) for v in p.volumes], combined=True, oriented=False)]
            for idx, port in enumerate(p.ports(stack), start=1):
                lo_, hi_ = p.bbox(stack)
                surfs = _find_port_surfaces(faces, port.center, port.normal, float(np.linalg.norm(hi_ - lo_)) / 2,
                                            tol=1e-3 * feature)
                gmsh.model.addPhysicalGroup(2, surfs, port_tag(p.id, idx), name=f"port_{p.id}_{port.name}")
                port_faces += surfs
        for c in conductors:
            vols = [(3, v) for n in c.nodes for v in prim_vols[n]]
            skin = [t for _, t in gmsh.model.getBoundary(vols, combined=True, oriented=False)]
            gmsh.model.addPhysicalGroup(2, skin, skin_tag(c.id), name=f"skin_{c.id}")
            ptag = lambda np_: port_tag(np_[0], by_id[np_[0]].port_index(np_[1], stack)) if np_ else None
            conductor_info.append({"id": c.id, "nodes": c.nodes, "skin": skin_tag(c.id),
                                   "port_hi": ptag(c.port_hi), "port_lo": ptag(c.port_lo)})

        # size field: fine at conductors, graded as growth × distance in the air (the shell is extruded afterwards)
        lc_min = feature / mesh.cells_across
        skins = [t for c in conductor_info for t in gmsh.model.getEntitiesForPhysicalGroup(2, c["skin"])]
        field = gmsh.model.mesh.field
        fd = field.add("Distance")
        field.setNumbers(fd, "SurfacesList", skins)
        field.setNumber(fd, "Sampling", 40)
        ft = field.add("Threshold")
        field.setNumber(ft, "InField", fd)
        field.setNumber(ft, "SizeMin", lc_min)
        field.setNumber(ft, "SizeMax", mesh.growth * r_in)
        field.setNumber(ft, "DistMin", lc_min / mesh.growth)
        field.setNumber(ft, "DistMax", r_in)
        fields = [ft]
        corner_faces = [t for p in primitives if p.type == "corner" for v in p.volumes
                        for _, t in gmsh.model.getBoundary([(3, v)], oriented=False)]
        if mesh.port_refine < 1.0 and (port_faces or corner_faces):
            fp = field.add("Distance")
            field.setNumbers(fp, "SurfacesList", sorted(set(port_faces + corner_faces)))
            field.setNumber(fp, "Sampling", 20)
            tp = field.add("Threshold")
            field.setNumber(tp, "InField", fp)
            field.setNumber(tp, "SizeMin", lc_min * mesh.port_refine)
            field.setNumber(tp, "SizeMax", mesh.growth * r_in)
            field.setNumber(tp, "DistMin", feature)
            field.setNumber(tp, "DistMax", feature + (mesh.growth * r_in - lc_min * mesh.port_refine) / mesh.growth)
            fields.append(tp)
        if background is not None:
            tets, sizes = background
            view = gmsh.view.add("zz_size")
            data = np.hstack([np.transpose(tets, (0, 2, 1)).reshape(len(tets), 12), sizes]).ravel()
            gmsh.view.addListData(view, "SS", len(tets), data.tolist())
            fv = field.add("PostView")
            field.setNumber(fv, "ViewTag", view)
            fields.append(fv)
        fmin = field.add("Min")
        field.setNumbers(fmin, "FieldsList", fields)
        field.setAsBackgroundMesh(fmin)
        for opt, val in (("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                         ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.Algorithm3D", mesh.algorithm3d)):
            gmsh.option.setNumber(opt, val)
        gmsh.model.mesh.generate(3)
        if mesh.optimize:
            gmsh.model.mesh.optimize("")        # gmsh's own tet optimizer
            gmsh.model.mesh.optimize("Netgen")  # Netgen smoothing/swapping removes most slivers
        n_tets = len(gmsh.model.mesh.getElementsByType(4)[0])
        if n_tets > mesh.max_tets:
            raise RuntimeError(f"mesh has {n_tets} tetrahedra > cap {mesh.max_tets}")
        _, qual = _qualities()
        if qual < mesh.min_quality:
            raise MeshQualityError(f"minimum SICN {qual:.4f} < {mesh.min_quality}")
        element_nodes = _element_nodes(primitives, prim_vols, air, stack, out_dir / f"{name}_element_nodes.npz",
                                       spacing=feature)
        n_prisms = _add_shell_layers(box_faces, center, r_in, r_out, mesh.shell_layers) if r_out else 0
        if n_tets + 3 * n_prisms > mesh.max_tets:
            raise RuntimeError(f"mesh has {n_tets} tetrahedra + {n_prisms} prisms (> {mesh.max_tets} tet-equivalents)")
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
        gmsh.option.setNumber("Mesh.ScalingFactor", UM)
        msh = out_dir / f"{name}.msh"
        gmsh.write(str(msh))
    finally:
        gmsh.finalize()

    info = ModelInfo(
        msh=str(msh), center_m=[float(v * UM) for v in center], r_in_m=r_in * UM,
        r_out_m=r_out * UM if r_out else None, n_tets=n_tets, n_prisms=n_prisms, n_nodes=n_nodes,
        mesh_time_s=time.perf_counter() - t0,
        volumes={p.id: stack.material_of(p.layer).sigma_S_per_m for p in primitives},
        conductors=conductor_info, eps_r_background=stack.materials[stack.background].eps_r,
        contacts=contacts, element_nodes=str(element_nodes), min_quality=qual,
    )
    info.save(out_dir / f"{name}.json")
    return info
