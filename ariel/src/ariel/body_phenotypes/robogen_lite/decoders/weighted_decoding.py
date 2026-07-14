"""Weighted high-probability decoder.

Variant of HighProbabilityDecoder where parent selection at decode step ``k``
is weighted by row ``k`` of a per-step weight matrix ``w_extend_matrix``.

At each decode step ``k = len(self.edges)``:
    1. For every instantiated parent p with at least one live face:
           parent_score[p] = w_extend_matrix[k, p] * max(conn_p_space[p] * available_faces[p])
    2. Pick the parent by argmax over parent_score.
    3. Within the chosen parent, pick (child, face) by argmax of its row.

Weights are used as raw values (no sigmoid). Only relative magnitudes within
a row matter for argmax, so any real-valued matrix works — including negative
entries, which suppress a module from being chosen at that step.

A parent whose faces are all used has row.max() == 0, so it is excluded from
the argmax regardless of its weight, and stops competing automatically.

Per-step rows let the genotype encode time-varying branching priorities
(e.g. "boost the core for the first few steps, then deepen one branch"),
while the decoder remains fully deterministic.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import DiGraph

from ariel import log
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleInstance,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)


class WeightedHighProbabilityDecoder(HighProbabilityDecoder):
    """High-probability decoder with per-step extension weights."""

    def probability_matrices_to_graph(
        self,
        type_probability_space: npt.NDArray[np.float32],
        connection_probability_space: npt.NDArray[np.float32],
        rotation_probability_space: npt.NDArray[np.float32],
        w_extend_matrix: npt.NDArray[np.float32],
    ) -> DiGraph[Any]:
        """
        Convert probability matrices + per-step weight matrix to a graph.

        Parameters
        ----------
        type_probability_space, connection_probability_space, rotation_probability_space
            Same as the parent class.
        w_extend_matrix
            Per-step, per-module weights, shape ``(num_modules, num_modules)``.
            Row ``k`` is used at decode step ``k`` (= number of edges already
            placed). Raw values; no sigmoid applied. Only relative magnitudes
            within a row matter for argmax.

        Returns
        -------
        DiGraph
            The decoded body graph.
        """
        # Reset graph
        self._graph: dict[int, ModuleInstance] = {}
        self.graph: DiGraph[Any] = nx.DiGraph()

        # Store probability spaces and weights
        self.conn_p_space = connection_probability_space.copy()
        self.rot_p_space = rotation_probability_space.copy()
        self.type_p_space = type_probability_space.copy()
        self.w_extend_matrix = np.asarray(w_extend_matrix, dtype=np.float64).copy()

        expected_shape = (self.num_modules, self.num_modules)
        if self.w_extend_matrix.shape != expected_shape:
            msg = (
                f"w_extend_matrix shape {self.w_extend_matrix.shape} does not "
                f"match expected {expected_shape}"
            )
            raise ValueError(msg)

        self.apply_connection_constraints()
        self.set_module_types_and_rotations()
        self.decode_probability_to_graph()
        self.generate_networkx_graph()
        return self.graph

    def decode_probability_to_graph(self) -> None:
        """Two-stage decode: pick parent via per-step weighted score, then pick child/face."""
        # Track which slots are instantiated (1) vs not (0)
        pre_nodes = dict.fromkeys(range(self.num_modules), 0)
        pre_nodes[IDX_OF_CORE] = 1
        pre_nodes = {
            i: v
            for i, v in pre_nodes.items()
            if self.type_dict[i] != ModuleType.NONE
        }

        self.edges = []
        self._init_spatial_state()

        available_faces = np.zeros_like(self.conn_p_space)
        available_faces[IDX_OF_CORE, :, :] = 1.0

        for _ in range(len(pre_nodes)):
            step = len(self.edges)
            w_row = self.w_extend_matrix[step]

            # Gate by face availability AND spatial occupancy: any
            # (parent, *, face) cell whose child target cell is already
            # occupied is zeroed out for this step.
            gated = self.conn_p_space * available_faces * self._spatial_mask()

            # ---- Stage 1: per-parent score = w_row[p] × best live cell of p ----
            # Non-live parents stay at -inf so they never win argmax, even when
            # all live parents have negative scores (raw genes can be negative).
            parent_scores = np.full(self.num_modules, -np.inf, dtype=np.float64)
            any_live = False
            for p in range(self.num_modules):
                if pre_nodes.get(p, 0) != 1:
                    continue  # parent not instantiated yet
                row = gated[p]
                m = row.max()
                if m == 0.0:
                    continue  # parent has no live faces (face-taken or spatially blocked)
                parent_scores[p] = w_row[p] * m
                any_live = True

            if not any_live:
                log.debug("No more connections can be made.")
                break

            chosen_parent = int(parent_scores.argmax())

            # ---- Stage 2: argmax (child, face) within chosen parent ----
            parent_row = gated[chosen_parent]
            flat_idx = int(parent_row.argmax())
            to_module, conn_face = np.unravel_index(flat_idx, parent_row.shape)
            to_module, conn_face = int(to_module), int(conn_face)
            from_module = chosen_parent

            # ---- Validation (same as parent class) ----
            if to_module == IDX_OF_CORE:
                msg = (
                    "Cannot connect to the core module as a child.\n"
                    "This indicates an error in decoding."
                )
                raise ValueError(msg)
            if self.type_dict[to_module] == ModuleType.NONE:
                msg = (
                    "Cannot instantiate a NONE module.\n"
                    "This indicates an error in decoding."
                )
                raise ValueError(msg)
            if self.type_dict[from_module] == ModuleType.NONE:
                msg = (
                    "Cannot instantiate a NONE module.\n"
                    "This indicates an error in decoding."
                )
                raise ValueError(msg)

            # ---- Commit edge and update state ----
            self.edges.append((from_module, to_module, conn_face))
            pre_nodes[to_module] = 1
            self.conn_p_space[from_module, :, conn_face] = 0.0
            self.conn_p_space[:, to_module, :] = 0.0
            available_faces[to_module, :, :] = 1.0
            self._record_attachment(from_module, to_module, conn_face)

        self.nodes = {i for i in pre_nodes if pre_nodes[i] == 1}
