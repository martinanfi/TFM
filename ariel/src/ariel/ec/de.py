import numpy as np
from ariel.ec.individual import Individual
from collections.abc import Sequence
import random
from ariel.simulation.controllers.cpg_decoder import CPGDecoder
from ariel.simulation.controllers.modulator import Modulator

GENE_SIZE_NDE = 64 #change later i guess
N_STATIC_FEATURES = 14   # rel_pos(3) + euclidean_dist(1) + depth(1) + xquat(4) + subtreemass(1) + joint_axis(3) + lateral_proj(1)
N_DYNAMIC_FEATURES = 16  # qpos(1) + qvel(1) + xvelp(3) + xvelr(3) + xquat(4)
N_FEATURES_TOTAL = N_STATIC_FEATURES + N_DYNAMIC_FEATURES
NUMELS_CONTROLLER = CPGDecoder(in_dim=N_STATIC_FEATURES, out_dim=3).num_params
NUMELS_MODULATOR = Modulator(dyn_dim=N_DYNAMIC_FEATURES, out_dim=3).num_params

from ariel.simulation.full_robot import process_genotype

type Population = Sequence[Individual]

N_MODULES = 30

class RevDE:

    def __init__(self, F, p, a, b, RNG = np.random.default_rng(42)):
        super().__init__()
        self.F = F
        self.p = p
        self.RNG = RNG
        self.a = a
        self.b = b

    def get_mutated_pop(self, pop : Population) -> Population:
        pop_size = len(pop) # store original size
        pop_indices = np.arange(pop_size)
        self.RNG.shuffle(pop_indices)

        pop_indices = [pop_indices, pop_indices[0:2]] # add first 2 to not loose inds
        pop_indices = np.concatenate(pop_indices)

        new_genotypes = []
        new_pop = []
        for start in range(0, len(pop_indices) - len(pop_indices) % 3, 3):
            i, j, k = pop_indices[start:start + 3]
            triplet = [pop[int(i)], pop[int(j)], pop[int(k)]]
            new_genotypes = matrix_transformation(triplet, self.F)
            for l in range(3):
                if len(new_pop) < pop_size:
                    ind = Individual()
                    ind.genotype = new_genotypes[l].tolist()
                    new_pop.append(ind)
        return new_pop[:pop_size]
    
    def crossover(self, original_pop : Population, mutated_pop : Population) -> Population:
        # v  = m ⊙ y_n + (1− m) ⊙ x_n.
        trial_pop = []
        for original_ind, mutated_ind in zip(original_pop, mutated_pop):
            x = np.asarray(original_ind.genotype)
            y = np.asarray(mutated_ind.genotype)

            m = self.RNG.random(x.shape) < self.p      # bool mask
            v = np.where(m, y, x)

            trial_ind = Individual()
            trial_ind.genotype = v.tolist()
            trial_pop.append(trial_ind)
        return trial_pop
    
    def one_point_crossover(self, ind1, ind2, only_morphology=True):

        genotype1 = process_genotype(
            gene_size_nde=GENE_SIZE_NDE,
            numels_controller=NUMELS_CONTROLLER,
            genotype_vec=ind1.genotype,
            n_modules=N_MODULES
        )

        genotype2 = process_genotype(
            gene_size_nde=GENE_SIZE_NDE,
            numels_controller=NUMELS_CONTROLLER,
            genotype_vec=ind2.genotype,
            n_modules=N_MODULES
        )

        gene_indexes = [
            int(self.RNG.integers(0, GENE_SIZE_NDE)),
            int(self.RNG.integers(0, N_MODULES)),
            int(self.RNG.integers(0, NUMELS_CONTROLLER)),
            int(self.RNG.integers(0, max(1, NUMELS_MODULATOR))),
        ]

        # i=0: body_genotype is a list [type_p, conn_p, rot_p] (each 1D, len=GENE_SIZE_NDE)
        # i=1: w_extend_matrix is 2D (N_MODULES, N_MODULES)
        # i=2: controller_tensor is 1D
        # i=3: modulator_tensor is 1D
        # only_morphology=True ⇒ crossover only morphology parts (body, w_extend);
        # inherit brain (controller, modulator) from parent 1.
        child_genes = []

        for i, (gene1, gene2, idx) in enumerate(
            zip(genotype1, genotype2, gene_indexes)
        ):
            do_crossover = (not only_morphology) or (i in (0, 1))

            if i == 0:
                # body_genotype: per-chromosome 1-point crossover at the same idx
                if do_crossover:
                    body_parts = [
                        np.concatenate([np.asarray(b1)[:idx], np.asarray(b2)[idx:]])
                        for b1, b2 in zip(gene1, gene2)
                    ]
                else:
                    body_parts = [np.asarray(b1).copy() for b1 in gene1]
                child_gene = np.concatenate(body_parts)
            else:
                arr1 = np.asarray(gene1)
                arr2 = np.asarray(gene2)
                if do_crossover:
                    # row-slice for 2D w_extend; element-slice for 1D tensors
                    child_gene = np.concatenate([arr1[:idx], arr2[idx:]], axis=0)
                else:
                    child_gene = arr1.copy()
                child_gene = child_gene.reshape(-1)

            child_genes.append(child_gene.astype(np.float32, copy=False))

        child_genotype = np.concatenate(child_genes)

        child = Individual()
        child.genotype = child_genotype.tolist()

        return child
            
    def select(
        self,
        original_pop: Population,
        evaluated_trial_pop: Population,
        maximistion: bool = True,
    ) -> Population:
        all_inds: list[Individual] = []

        if maximistion:
            for i in range(len(original_pop)):
                original_ind = original_pop[i]
                trial_ind = evaluated_trial_pop[i]

                if original_ind.fitness >= trial_ind.fitness:
                    # original survives, trial dies
                    original_ind.alive = True
                    trial_ind.alive = False
                else:
                    # trial survives, original dies
                    original_ind.alive = False
                    trial_ind.alive = True

                # IMPORTANT: add *both* so DB sees their updated .alive
                all_inds.extend([original_ind, trial_ind])
        else:
            for i in range(len(original_pop)):
                original_ind = original_pop[i]
                trial_ind = evaluated_trial_pop[i]

                if original_ind.fitness <= trial_ind.fitness:
                    # original survives, trial dies
                    original_ind.alive = True
                    trial_ind.alive = False
                else:
                    original_ind.alive = False
                    trial_ind.alive = True

                all_inds.extend([original_ind, trial_ind])

        return all_inds

    def select_ldelta(
        self,
        original_pop: Population,
        evaluated_trial_pop: Population,
        maximistion: bool = True,
    ) -> Population:
        import math
        all_inds: list[Individual] = []

        a = self.a
        b = self.b

        def _score(ind):
            fit = ind.fitness
            # Non-finite fitness ⇒ uniformly worst; never wins selection by NaN propagation.
            if not math.isfinite(fit):
                return -float("inf")
            pre = ind.fitness_pre
            # No valid pre-CMA reference ⇒ no ldelta credit.
            if not math.isfinite(pre):
                return a * fit
            return a * fit + b * (fit - pre)

        final_score_original_pop = [_score(ind) for ind in original_pop]
        final_score_trial_pop    = [_score(ind) for ind in evaluated_trial_pop]

        if maximistion:
            for i in range(len(original_pop)):
                original_ind = original_pop[i]
                original_score = final_score_original_pop[i]
                trial_ind = evaluated_trial_pop[i]
                trial_score = final_score_trial_pop[i]

                if original_score >= trial_score:
                    # original survives, trial dies
                    original_ind.alive = True
                    trial_ind.alive = False
                else:
                    # trial survives, original dies
                    original_ind.alive = False
                    trial_ind.alive = True

                # IMPORTANT: add *both* so DB sees their updated .alive
                all_inds.extend([original_ind, trial_ind])
        else:
            for i in range(len(original_pop)):
                original_ind = original_pop[i]
                original_score = final_score_original_pop[i]
                trial_ind = evaluated_trial_pop[i]
                trial_score = final_score_trial_pop[i]

                if original_score <= trial_score:
                    # original survives, trial dies
                    original_ind.alive = True
                    trial_ind.alive = False
                else:
                    original_ind.alive = False
                    trial_ind.alive = True

                all_inds.extend([original_ind, trial_ind])

        return all_inds

# Genome block layout, must match the slicing in `process_genotype`.
# Used by `matrix_transformation` when F is a per-block dict so each slice
# of the genome can be perturbed at its own scale.
BLOCK_SIZES = {
    "body":       3 * GENE_SIZE_NDE,
    "w_extend":   N_MODULES * N_MODULES,
    "controller": NUMELS_CONTROLLER,
    "modulator":  NUMELS_MODULATOR,
}


def _rev_matrix(F: float) -> np.ndarray:
    return np.array([
        [1, F, -F],
        [-F, 1 - F**2, F + F**2],
        [F + F**2, -F + F**2 + F**3, 1 - 2 * F**2 - F**3],
    ])


def matrix_transformation(triplet, F):
    """RevDE triplet transform; F may be a scalar or per-block dict.

    When F is a dict it must have a key for every entry in BLOCK_SIZES;
    each genome block is transformed by its own matrix. The total length
    of BLOCK_SIZES must match the genome length.
    """
    g = np.stack([individual.genotype for individual in triplet], axis=0)

    if isinstance(F, dict):
        missing = set(BLOCK_SIZES) - set(F)
        if missing:
            msg = f"F dict is missing keys: {sorted(missing)}"
            raise KeyError(msg)
        out = np.empty_like(g, dtype=np.float64)
        ptr = 0
        for name, size in BLOCK_SIZES.items():
            out[:, ptr:ptr + size] = _rev_matrix(F[name]) @ g[:, ptr:ptr + size]
            ptr += size
        if ptr != g.shape[1]:
            msg = (
                f"BLOCK_SIZES sums to {ptr} but genome length is {g.shape[1]}"
            )
            raise ValueError(msg)
        return out

    return _rev_matrix(F) @ g
