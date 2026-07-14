"""Highest-probability-decoding algorithm for ARIEL-robots.

Todo
----
- [ ] for loops to be replaced with vectorized operations
"""

# Evaluate type annotations in a deferred manner (ruff: UP037)
from __future__ import annotations

# Standard library
from typing import Any

# Third-party libraries
import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import DiGraph

# Local libraries
from ariel import log
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    ModuleFaces,
    ModuleInstance,
    ModuleRotationsIdx,
    ModuleType,
)

# Third-party libraries
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import Blueprint

# --- Spatial-occupancy bookkeeping ---------------------------------------- #
# Each module is built with its geom extending along +y_local from the body
# origin (which sits at the module's BACK face). On a unit integer grid in
# the core's frame, a child placed via face F of a parent therefore lands at
# parent.pos + parent.R @ _FACE_LOCAL_DIR[F]; the child's local-to-world
# rotation propagates as child.R = parent.R @ _R_FACE[F]. Per-module 45°/90°
# rotations are NOT modeled here — they don't change the child's grid cell,
# only its own local frame around the outward axis.
_FACE_LOCAL_DIR: dict[ModuleFaces, np.ndarray] = {
    ModuleFaces.FRONT:  np.array([ 0,  1,  0], dtype=np.int64),
    ModuleFaces.BACK:   np.array([ 0, -1,  0], dtype=np.int64),
    ModuleFaces.LEFT:   np.array([-1,  0,  0], dtype=np.int64),
    ModuleFaces.RIGHT:  np.array([ 1,  0,  0], dtype=np.int64),
    ModuleFaces.TOP:    np.array([ 0,  0,  1], dtype=np.int64),
    ModuleFaces.BOTTOM: np.array([ 0,  0, -1], dtype=np.int64),
}
_R_FACE: dict[ModuleFaces, np.ndarray] = {
    ModuleFaces.FRONT:  np.eye(3, dtype=np.int64),
    ModuleFaces.BACK:   np.array([[-1, 0, 0], [ 0, -1, 0], [ 0,  0, 1]], dtype=np.int64),
    ModuleFaces.LEFT:   np.array([[ 0,-1, 0], [ 1,  0, 0], [ 0,  0, 1]], dtype=np.int64),
    ModuleFaces.RIGHT:  np.array([[ 0, 1, 0], [-1,  0, 0], [ 0,  0, 1]], dtype=np.int64),
    ModuleFaces.TOP:    np.array([[ 1, 0, 0], [ 0,  0,-1], [ 0,  1, 0]], dtype=np.int64),
    ModuleFaces.BOTTOM: np.array([[ 1, 0, 0], [ 0,  0, 1], [ 0, -1, 0]], dtype=np.int64),
}


class HighProbabilityDecoder(Blueprint):
    """Implements the high-probability-decoding algorithm."""

    def probability_matrices_to_graph(
        self,
        type_probability_space: npt.NDArray[np.float32],
        connection_probability_space: npt.NDArray[np.float32],
        rotation_probability_space: npt.NDArray[np.float32],
    ) -> DiGraph[Any]:
        """
        Convert probability matrices to a graph.

        Parameters
        ----------
        type_probability_space
            Probability space for module types.
        connection_probability_space
            Probability space for connections between modules.
        rotation_probability_space
            Probability space for module rotations.

        Returns
        -------
        DiGraph
            A graph representing the decoded modules and their connections.
        """
        # Reset the graph
        self._graph: dict[int, ModuleInstance] = {}
        self.graph: DiGraph[Any] = nx.DiGraph()

        # Store the probability spaces
        self.conn_p_space = connection_probability_space.copy()
        self.rot_p_space = rotation_probability_space.copy()
        self.type_p_space = type_probability_space.copy()

        # Apply constraints
        self.apply_connection_constraints()

        # Initialize module types and rotations
        self.set_module_types_and_rotations()

        # Decode probability spaces into a graph
        self.decode_probability_to_graph()

        # Create the final graph from the simple graph
        self.generate_networkx_graph()
        return self.graph

    def _init_spatial_state(self) -> None:
        """Initialize per-module pose tracking for spatial-occupancy checks."""
        self.module_pos: dict[int, tuple[int, int, int]] = {
            IDX_OF_CORE: (0, 0, 0),
        }
        self.module_R: dict[int, np.ndarray] = {
            IDX_OF_CORE: np.eye(3, dtype=np.int64),
        }
        self.occupied_coords: dict[tuple[int, int, int], int] = {
            (0, 0, 0): IDX_OF_CORE,
        }

    def _spatial_mask(self) -> np.ndarray:
        """Return a `conn_p_space`-shaped mask zeroing `(parent, *, face)`
        cells whose target child cell is already occupied. Ones elsewhere."""
        mask = np.ones_like(self.conn_p_space, dtype=np.float64)
        for parent_idx, parent_pos in self.module_pos.items():
            parent_R = self.module_R[parent_idx]
            ppos = np.asarray(parent_pos, dtype=np.int64)
            for face, dir_local in _FACE_LOCAL_DIR.items():
                world_dir = parent_R @ dir_local
                child_pos = tuple(int(c) for c in (ppos + world_dir))
                if child_pos in self.occupied_coords:
                    mask[parent_idx, :, face.value] = 0.0
        return mask

    def _record_attachment(
        self,
        from_module: int,
        to_module: int,
        conn_face: int,
    ) -> None:
        """Record the new child's grid cell and orientation."""
        face = ModuleFaces(conn_face)
        parent_pos = np.asarray(self.module_pos[from_module], dtype=np.int64)
        parent_R = self.module_R[from_module]
        child_pos = tuple(
            int(c) for c in (parent_pos + parent_R @ _FACE_LOCAL_DIR[face])
        )
        self.module_pos[to_module] = child_pos
        self.module_R[to_module] = parent_R @ _R_FACE[face]
        self.occupied_coords[child_pos] = to_module

    def decode_probability_to_graph(
        self,
    ) -> None:
        """
        Decode the probability spaces into a graph.

        Raises
        ------
        ValueError
            If an attempt is made to use the core module as a child.
        ValueError
            If an attempt is made to instantiate a NONE module as a parent.
        ValueError
            If an attempt is made to instantiate a NONE module as a child.
        """
        # Create a dictionary to track instantiated modules
        pre_nodes = dict.fromkeys(range(self.num_modules), 0)

        # The core module is always instantiated
        pre_nodes[IDX_OF_CORE] = 1

        # Remove 'None' modules from instantiated modules
        pre_nodes = {
            i: v
            for i, v in pre_nodes.items()
            if self.type_dict[i] != ModuleType.NONE
        }

        # List to hold edges (parent, child, face)
        self.edges = []
        self._init_spatial_state()

        # Available faces for connections
        available_faces = np.zeros_like(self.conn_p_space)
        available_faces[IDX_OF_CORE, :, :] = 1.0
        for _ in range(len(pre_nodes)):
            # Get the current state of the connection probability space,
            # gated by both face availability and spatial occupancy.
            current_state = (
                self.conn_p_space * available_faces * self._spatial_mask()
            )

            # Find the maximum value in the connection probability space
            max_index = np.unravel_index(
                np.argmax(current_state),
                current_state.shape,
            )

            # Convert to list for easier manipulation
            max_index = [int(i) for i in max_index]
            from_module, to_module, conn_face = max_index

            # Check if the maximum value is zero (no more connections)
            value_at_max = current_state[from_module, to_module, conn_face]
            if value_at_max == 0.0:
                msg = "No more connections can be made."
                log.debug(msg)
                break

            # Ensure the core module is never a child
            if to_module == IDX_OF_CORE:
                msg = "Cannot connect to the core module as a child.\n"
                msg += "This indicates an error in decoding."
                raise ValueError(msg)

            # Ensure no NONE modules are instantiated
            if self.type_dict[to_module] == ModuleType.NONE:
                msg = "Cannot instantiate a NONE module.\n"
                msg += "This indicates an error in decoding."
                raise ValueError(msg)

            if self.type_dict[from_module] == ModuleType.NONE:
                msg = "Cannot instantiate a NONE module.\n"
                msg += "This indicates an error in decoding."
                raise ValueError(msg)

            # Get module types and rotations
            self.edges.append(
                (from_module, to_module, conn_face),
            )

            # Update instantiated modules
            pre_nodes[to_module] = 1

            # Disable taken face
            self.conn_p_space[from_module, :, conn_face] = 0.0

            # Child has only one parent
            self.conn_p_space[:, to_module, :] = 0.0

            # Update available faces
            available_faces[to_module, :, :] = 1.0
            self._record_attachment(from_module, to_module, conn_face)

        # Nodes and edges of the final graph
        self.nodes = {i for i in pre_nodes if pre_nodes[i] == 1}

    def set_module_types_and_rotations(self) -> None:
        """Set the module types and rotations using probability spaces."""
        # Module type from argmax of type probability space
        type_from_argmax = np.argmax(self.type_p_space, axis=1)
        self.type_dict = {
            i: ModuleType(int(type_from_argmax[i]))
            for i in range(self.num_modules)
        }

        # Constrain rotations and connections based on module types
        all_possible_faces = set(ModuleFaces)
        all_possible_rotations = set(ModuleRotationsIdx)
        for module_idx, module_type in self.type_dict.items():
            # NONE modules cannot participate as parent or child at all.
            # Previously this was achieved as a side effect of the
            # "disable as child by face" line below, which was removed
            # because it incorrectly conflated the parent-face axis with
            # child-face semantics and blocked the core's BACK face.
            if module_type == ModuleType.NONE:
                self.conn_p_space[module_idx, :, :] = 0.0
                self.conn_p_space[:, module_idx, :] = 0.0

            # Constrain connections based on module type
            allowed_faces = set(ALLOWED_FACES[module_type])
            disallowed_faces = all_possible_faces - allowed_faces
            for face in disallowed_faces:
                # Disable as parent
                self.conn_p_space[module_idx, :, face.value] = 0.0

            # Constrain rotations based on module type
            allowed_rotations = set(ALLOWED_ROTATIONS[module_type])
            disallowed_rotations = all_possible_rotations - allowed_rotations
            for rotation in disallowed_rotations:
                self.rot_p_space[module_idx, rotation.value] = 0.0

        # Rotation type form argmax of rotation probability space
        rot_from_argmax = np.argmax(self.rot_p_space, axis=1)
        self.rot_dict = {
            i: ModuleRotationsIdx(int(rot_from_argmax[i]))
            for i in range(self.num_modules)
        }

    def apply_connection_constraints(
        self,
    ) -> None:
        """Apply connection constraints to probability spaces."""
        # Self connection not allowed
        for i in range(NUM_OF_FACES):
            np.fill_diagonal(self.conn_p_space[:, :, i], 0.0)

        # Core is unique
        self.type_p_space[:, int(ModuleType.CORE.value)] = 0.0
        self.type_p_space[IDX_OF_CORE, int(ModuleType.CORE.value)] = 1.0

        # Core is always a parent, never a child
        self.conn_p_space[:, IDX_OF_CORE, :] = 0.0
