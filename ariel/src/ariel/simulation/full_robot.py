import mujoco
import networkx as nx
import math
import torch
import numpy as np
from dataclasses import dataclass
import sys
sys.path.insert(0, "/Users/anfi/Documents/TFM_FINAL/ariel/src")

from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency, NaCPG
from ariel.simulation.controllers.cpg_decoder import CPGDecoder
from ariel.simulation.controllers.modulator import Modulator
from ariel.simulation.environments import SimpleFlatWorld
from ariel.ec.individual import Individual
from typing import Any
from enum import Enum

from ariel.body_phenotypes.robogen_lite.decoders.weighted_decoding import WeightedHighProbabilityDecoder as whpd
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder as hpd
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding as nde
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

GENE_SIZE_NDE = 64 #change later i guess
N_STATIC_FEATURES = 14   # rel_pos(3) + euclidean_dist(1) + depth(1) + xquat(4) + subtreemass(1) + joint_axis(3) + lateral_proj(1)
N_DYNAMIC_FEATURES = 16  # qpos(1) + qvel(1) + xvelp(3) + xvelr(3) + xquat(4)
N_FEATURES_TOTAL = N_STATIC_FEATURES + N_DYNAMIC_FEATURES
NUMELS_CONTROLLER = CPGDecoder(in_dim=N_STATIC_FEATURES, out_dim=3).num_params
CPG_PARAM_SCALE = 2 * math.pi   # matches NaCPG init range for phase/amp/w
# amplitude column is bounded by the actuator hard clamp (±π/2) — letting the
# decoder reach 2π means most evolved amplitudes saturate the clamp and the
# CPG output becomes a square wave instead of a sinusoid.
CPG_AMP_SCALE = math.pi / 2


@dataclass
class full_robot:
    body_phenotype: nx.DiGraph
    cpg_decoder: CPGDecoder
    modulator: Modulator
    controller: NaCPG
    actuators: list[int]  # graph node ids, in actuator-index order; data.ctrl[i] drives actuators[i]
    core: object
    world: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    core_id: int
    forward_vec: np.ndarray
    node_to_body_name: dict[int, str]
    core_pos: np.ndarray
    static_features: np.ndarray  # (n_actuators, N_STATIC_FEATURES), computed once at decode

    def set_forward_vec(self):
        """Return the current forward direction (2D ground plane) from live sim data."""
        quat = self.data.xquat[self.core_id]
        forward_world = np.zeros(3)
        mujoco.mju_rotVecQuat(forward_world, np.array([1.0, 0.0, 0.0]), quat)
        self.forward_vec = forward_world[:2]

    def get_alpha_angle(self, target_direction_deg):
        heading_rad = np.arctan2(self.forward_vec[1], self.forward_vec[0])
        target_rad = np.deg2rad(target_direction_deg)
        alpha = target_rad - heading_rad
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        return float(alpha)

    def get_dynamic_features(self, target_direction_deg) -> np.ndarray:
        """Per-timestep features. Returns (n_actuators, N_DYNAMIC_FEATURES)."""
        alpha_angle = self.get_alpha_angle(target_direction_deg)
        alpha_arr = np.array(
            [np.sin(alpha_angle), np.cos(alpha_angle)], dtype="float32",
        )
        feats = []
        for cpg_idx, node_id in enumerate(self.actuators):
            body_name = self.node_to_body_name[node_id]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = self.model.actuator_trnid[cpg_idx, 0]

            qpos = np.array([self.data.qpos[self.model.jnt_qposadr[joint_id]]])
            qvel = np.array([self.data.qvel[self.model.jnt_dofadr[joint_id]]])
            cvel = self.data.cvel[body_id]   # [ang(3), lin(3)] at CoM, world frame
            xvelr = cvel[0:3].copy()
            xvelp = cvel[3:6].copy()
            xquat = self.data.xquat[body_id].copy()

            feat = np.concatenate(
                [qpos, qvel, xvelp, xvelr, xquat, alpha_arr, self.forward_vec],
            )
            feats.append(feat)

        return np.stack(feats, axis=0).astype("float32")
    
    def set_static_features(self) -> None:
        """Recompute and store static features in-place. Sets self.static_features."""
        depths = nx.single_source_shortest_path_length(self.body_phenotype, 0)

        fwd = self.forward_vec[:2]
        norm = np.linalg.norm(fwd)
        lateral_unit = (
            np.array([fwd[1], -fwd[0]]) / norm if norm > 1e-8 else np.zeros(2)
        )

        feats = []
        for cpg_idx, node_id in enumerate(self.actuators):
            body_name = self.node_to_body_name[node_id]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = self.model.actuator_trnid[cpg_idx, 0]

            rel_pos = self.data.xpos[body_id] - self.core_pos
            euclidean_dist = np.linalg.norm(rel_pos)
            depth = float(depths.get(node_id, 0))
            xquat = self.data.xquat[body_id].copy()
            subtreemass = float(self.model.body_subtreemass[body_id])
            joint_axis = self.model.jnt_axis[joint_id].copy()
            lateral_proj = float(np.dot(rel_pos[:2], lateral_unit))

            feat = np.concatenate(
                [rel_pos, [euclidean_dist], [depth], xquat,
                 [subtreemass], joint_axis, [lateral_proj]],
            )
            feats.append(feat)

        self.static_features = np.stack(feats, axis=0).astype("float32")


def decode_robot(individual: Individual, NDE, DECODER, world_cls = SimpleFlatWorld, seed: int = 0) -> full_robot:
    """
    From an Individual (body + controller genotype) produce a fully
    decoded robot + simulation, ready for evaluation.
    """

    # ---------------- BODY: NDE -> graph ----------------
    n_modules = DECODER.num_modules
    body_genotype, w_extend_matrix, controller_genotype, modulator_genotype = process_genotype(
        GENE_SIZE_NDE, NUMELS_CONTROLLER, individual.genotype, n_modules,
    )

    probabilities = NDE.forward(body_genotype)
    body_phenotype = DECODER.probability_matrices_to_graph(
        probabilities[0],
        probabilities[1],
        probabilities[2],
        w_extend_matrix,
    )

    # ---------------- SIM BUILD ----------------
    (
        core,
        world,
        model,
        data,
        modules,
        core_id,
        forward_vec,
        node_to_body_name,
        core_pos,
    ) = build_world(body_phenotype, world_cls=world_cls)

    # List of actuator node ids (hinge modules)
    actuators = [
        node_id
        for node_id, attrs in body_phenotype.nodes(data=True)
        if attrs["type"] == ModuleType.HINGE.name
    ]
    if len(actuators) == 0:
        return full_robot(
            body_phenotype=body_phenotype,
            cpg_decoder=None,
            modulator=None,
            controller=None,
            actuators=[],
            core=core,
            world=world,
            model=model,
            data=data,
            core_id=core_id,
            forward_vec=forward_vec,
            node_to_body_name=node_to_body_name,
            core_pos=core_pos,
            static_features=None
        )

    static_features = get_static_features(
        body_phenotype=body_phenotype,
        actuators=actuators,
        node_to_body_name=node_to_body_name,
        model=model,
        data=data,
        core_pos=core_pos,
        forward_vec=forward_vec,
    )

    # ---------------- CONTROLLER: genotype -> transformer weights ----------------

    cpg_decoder = CPGDecoder(
        in_dim=N_STATIC_FEATURES,
        out_dim=3,
        param_scales=(CPG_PARAM_SCALE, CPG_AMP_SCALE, CPG_PARAM_SCALE),
    )
    cpg_decoder.set_flat_params(controller_genotype)

    modulator = Modulator(
        dyn_dim=N_DYNAMIC_FEATURES,
    )
    modulator.set_flat_params(modulator_genotype)

    cpg_params = cpg_decoder.forward(torch.from_numpy(static_features))  # [N, 3] = phase, amplitudes, w

    adjacency_dict = create_fully_connected_adjacency(len(actuators))
    controller = NaCPG(
        adjacency_dict,
        base_params=cpg_params,
        seed=seed,
    )
    controller.set_param_with_dict({
        "phase":      cpg_params[:, 0],
        "amplitudes": cpg_params[:, 1],
        "w":          cpg_params[:, 2],
    })

    return full_robot(
        body_phenotype=body_phenotype,
        cpg_decoder=cpg_decoder,
        modulator=modulator,
        controller=controller,
        actuators=actuators,
        core=core,
        world=world,
        model=model,
        data=data,
        core_id=core_id,
        forward_vec=forward_vec,
        node_to_body_name=node_to_body_name,
        core_pos=core_pos,
        static_features=static_features,
    )

def update_forward_vec(model, data, core_id):
    """Return the current forward direction (2D ground plane) from live sim data."""
    quat = data.xquat[core_id]
    forward_world = np.zeros(3)
    mujoco.mju_rotVecQuat(forward_world, np.array([1.0, 0.0, 0.0]), quat)
    return forward_world[:2]

def build_world(body_phenotype, world_cls):

    core = construct_mjspec_from_graph(body_phenotype)

    world = world_cls()
    world.spawn(core.spec)

    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    modules = core.modules  # dict[int, Module]

    node_to_body_name: dict[int, str] = {}
    for node_id, module in modules.items():
        if module is None:
            continue
        if isinstance(module, CoreModule):
            body_name = "core"
        else:
            body_name = module.body.name
        node_to_body_name[node_id] = body_name

    core_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")
    core_pos = data.xpos[core_body_id].copy()
    core_id = core_body_id

    quat = data.xquat[core_id]
    forward_world = np.zeros(3)
    mujoco.mju_rotVecQuat(forward_world, np.array([1.0, 0.0, 0.0]), quat)
    forward_vec = forward_world[:2]

    return core, world, model, data, modules, core_id, forward_vec, node_to_body_name, core_pos



def get_coords_mujoco(
    model,
    data,
    node_to_body_name: dict[int, str],
):
    """
    returns core_id for evaluation purposes
    Compute integer 3D lattice coordinates for each module from a built MuJoCo model.

    Parameters
    ----------
    model : mujoco.MjModel
        Compiled MuJoCo model.
    data : mujoco.MjData
        Runtime data (must be after mj_forward / sim.step(0) so xpos is valid).
    node_to_body_name : dict[int, str]
        Mapping from your node id (graph node) to the corresponding MuJoCo body name.
    module_spacing : float, optional
        Distance in MuJoCo units between centers of adjacent modules in the lattice.
        (E.g. if you placed modules exactly 1.0 apart, leave as 1.0.)
    core_id : int, optional
        Node id of the core. If None, assumes the lowest node id is the core.

    Returns
    -------
    coords : dict[int, np.ndarray(shape=(4,))]
        Mapping node_id -> [x_int, y_int, z_int, manhattan_distance]
        where (0,0,0) is the core.
    """
    core_id = None
    coords: dict[int, np.ndarray] = {}

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    
    for node_id, body_name in node_to_body_name.items():
        if node_id == 0: # when the node is the core
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names[3]) #core_id is always 3, u can check in names
            core_id = body_id
            core_pos = data.xpos[body_id]
        else:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # Absolute world position of this module's body center
        pos_world = data.xpos[body_id]

        # Position relative to core, then normalize by module spacing
        rel_pos = pos_world - core_pos

        # Manhattan distance from core in lattice space
        euclidean_dist = np.abs(rel_pos).sum()

        # Final feature vector (x, y, z, euclidean)
        feat = np.empty(4, dtype=float)
        feat[:3] = rel_pos
        feat[3] = euclidean_dist

        coords[node_id] = feat

    return coords, core_id

def process_genotype(gene_size_nde, numels_controller, genotype_vec, n_modules, weighted = True):
    """Slice the flat genotype.

    Layout (in order):
        [0 : 3*gene_size_nde]                                    body genes (type/conn/rot chromosomes)
        [3*gene_size_nde : 3*gene_size_nde + n_modules**2]        w_extend matrix (per-step × per-module, only if weighted)
        [... : ... + numels_controller]                           controller params
        [...:]                                                    modulator params
    """
    flat = np.asarray(genotype_vec)

    # body parts (NDE input chromosomes)
    type_p = flat[0:gene_size_nde].astype("float32", copy=True)
    conn_p = flat[gene_size_nde:2 * gene_size_nde].astype("float32", copy=True)
    rot_p  = flat[2 * gene_size_nde:3 * gene_size_nde].astype("float32", copy=True)

    body_genotype = [type_p, conn_p, rot_p]

    cursor = 3 * gene_size_nde

    if weighted:
        w_end = cursor + n_modules * n_modules
        w_extend_matrix = (
            flat[cursor:w_end]
            .astype("float32", copy=True)
            .reshape(n_modules, n_modules)
        )
        cursor = w_end

    # brain / controller part
    controller_flat = flat[cursor : cursor + numels_controller]
    controller_tensor = torch.from_numpy(
        controller_flat.astype("float32", copy=False)
    )
    cursor += numels_controller

    modulator_flat = flat[cursor:]
    modulator_tensor = torch.from_numpy(
        modulator_flat.astype("float32", copy=False)
    )

    if weighted:
        return [body_genotype, w_extend_matrix, controller_tensor, modulator_tensor]
    return [body_genotype, controller_tensor, modulator_tensor]

def get_static_features(
    body_phenotype,
    actuators,
    node_to_body_name,
    model,
    data,
    core_pos,
    forward_vec,
):
    """Called once at decode time. Returns (n_actuators, N_STATIC_FEATURES)."""
    depths = nx.single_source_shortest_path_length(body_phenotype, 0)

    # Robot-frame lateral unit vector (90° clockwise from forward).
    # Used to project rel_pos onto the side axis: + right, − left.
    fwd = forward_vec[:2]
    norm = np.linalg.norm(fwd)
    lateral_unit = (
        np.array([fwd[1], -fwd[0]]) / norm if norm > 1e-8 else np.zeros(2)
    )

    feats = []
    for cpg_idx, node_id in enumerate(actuators):
        body_name = node_to_body_name[node_id]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = model.actuator_trnid[cpg_idx, 0]

        rel_pos = data.xpos[body_id] - core_pos
        euclidean_dist = np.linalg.norm(rel_pos)
        depth = float(depths.get(node_id, 0))
        xquat = data.xquat[body_id].copy()
        subtreemass = float(model.body_subtreemass[body_id])
        joint_axis = model.jnt_axis[joint_id].copy()
        lateral_proj = float(np.dot(rel_pos[:2], lateral_unit))

        feat = np.concatenate(
            [rel_pos, [euclidean_dist], [depth], xquat,
             [subtreemass], joint_axis, [lateral_proj]],
        )
        feats.append(feat)

    return np.stack(feats, axis=0).astype("float32")

if __name__ == '__main__':
    from ariel.ec.de import RevDE

    SEED = 42
    RNG = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    F = 0.5 # scaling factor DE
    P = 0.7 # probability for DE mask

    # weighted sum for ldelta aware selection
    A = 0.5 # fitness weight
    B = 0.5 # ldelta weight
    NMODULES = 30
    GENE_SIZE_NDE = 64
    NUMELS_MODULATOR = Modulator(
        dyn_dim=N_DYNAMIC_FEATURES,
    ).num_params

    NDE = nde(number_of_modules=NMODULES, genotype_size=GENE_SIZE_NDE)
    DECODER = hpd(num_modules=NMODULES)
    DE = RevDE(F, P, A, B, RNG=RNG)

    def create_individual(RNG, n_genes_body=64, n_genes_brain=NUMELS_CONTROLLER + NUMELS_MODULATOR, n_modules=NMODULES) -> Individual:
        ind = Individual()
        range_probs = 5
        type_p_genes = RNG.uniform(-range_probs, range_probs, size=n_genes_body)
        conn_p_genes = RNG.uniform(-range_probs, range_probs, size=n_genes_body)
        rot_p_genes  = RNG.uniform(-range_probs, range_probs, size=n_genes_body)
        # per-step × per-module extension weights (raw; no sigmoid)
        w_extend_genes = RNG.uniform(-2.0, 2.0, size=n_modules * n_modules)
        # controller in [-1, 1]
        controller_genes = RNG.uniform(
            low=-1.0,
            high=1.0,
            size=n_genes_brain,
        ).astype("float32")
        flat_genotype = np.concatenate(
            [type_p_genes, conn_p_genes, rot_p_genes, w_extend_genes, controller_genes]
        ).astype("float32")
        ind.genotype = flat_genotype.tolist()
        return ind
    
    ind_example = create_individual(RNG)
    robot = decode_robot(individual=ind_example, NDE=NDE, DECODER=DECODER)



