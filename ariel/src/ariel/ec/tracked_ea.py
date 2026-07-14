"""TrackedEA: EA with per-generation CSV logging and generation timing."""

import csv
import json
import time
from pathlib import Path
from typing import Any

from rich.progress import track

from ariel.ec.ea import EA, EAOperation, config
from ariel.ec.population import Population


def log_alive_population_csv(
    alive_pop: Population,
    generation: int,
    csv_path: Path,
) -> None:
    """Append one row per individual to ``csv_path``.

    Writes the column header if the file is empty or does not yet exist.
    Adds a ``.csv`` suffix to ``csv_path`` if missing. Caller is
    responsible for filtering ``alive_pop`` to the individuals to log.

    Parameters
    ----------
    alive_pop : Population
        Individuals to log.
    generation : int
        Generation number recorded on each row.
    csv_path : Path
        Destination CSV path. Parent directories are created if missing.
    """
    csv_path = Path(csv_path).with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(
                # Trailing columns appended (not inserted) so existing readers that
                # index the first 7 positionally keep working. fitness/fitness_pre
                # are the SHAPED (selection) scores; *_raw are the undiscounted task
                # performance and collision_duty[_pre] the self-collision fraction,
                # populated from tags when the driver records them (blank otherwise).
                ["generation", "fitness", "fitness_pre", "genotype", "cpg_params", "learning_curve", "measures",
                 "fitness_raw", "fitness_pre_raw", "collision_duty", "collision_duty_pre"],
            )
        for ind in alive_pop:
            cpg = ind.CPG_params_ if ind.CPG_params_ is not None else []
            tags = ind.tags_ or {}
            curve = tags.get("learning_curve", [])
            measures = tags.get("measures", [])
            writer.writerow(
                [
                    generation,
                    ind.fitness,
                    ind.fitness_pre,
                    json.dumps(ind.genotype_),
                    json.dumps(cpg),
                    json.dumps(curve),
                    json.dumps(measures),
                    tags.get("fitness_raw", ""),
                    tags.get("fitness_pre_raw", ""),
                    tags.get("collision_duty", ""),
                    tags.get("collision_duty_pre", ""),
                ],
            )


class TrackedEA(EA):
    """``EA`` subclass that logs the alive population to CSV each generation.

    Adds per-generation timing output and constructs the CSV filename from
    experiment metadata. Filename layout:

    - ``ldelta{seed}_{pop_size}-{num_steps}_{learning_epochs}_{tag}.csv``
    - ``learning{seed}_{pop_size}-{num_steps}_{learning_epochs}_{tag}.csv``
    - ``evo{seed}_{pop_size}-{num_steps}_{tag}.csv``

    Parameters
    ----------
    population : Population
        Initial population. Forwarded to ``EA``.
    operations : list[EAOperation]
        Operation pipeline. Forwarded to ``EA``.
    pop_size : int
        Target population size. Embedded in the CSV filename.
    learning : bool
        Whether learning is applied during evaluation.
    ldelta : bool
        Whether learning uses the delta variant. Selects filename prefix
        (``ldelta`` vs ``learning`` vs ``evo``).
    learning_epochs : int
        Number of learning epochs per evaluation. Embedded in the filename
        when ``learning`` is True.
    tag : str
        Free-form experiment tag used as filename suffix.
    seed : int
        RNG seed; embedded in the filename for reproducibility tracking.
    log_dir : Path, optional
        Output directory for CSVs. Default ``Path("experiment_data")``.
    **ea_kwargs
        Forwarded to ``EA.__init__`` (e.g. ``num_steps``, ``db_file_path``).
    """

    def __init__(
        self,
        population: Population,
        operations: list[EAOperation],
        *,
        pop_size: int,
        learning: bool,
        ldelta: bool,
        learning_epochs: int,
        tag: str,
        seed: int,
        log_dir: Path = Path("experiment_data"),
        **ea_kwargs: Any,
    ) -> None:
        super().__init__(population, operations, **ea_kwargs)
        self.pop_size = pop_size
        self.learning = learning
        self.ldelta = ldelta
        self.learning_epochs = learning_epochs
        self.tag = tag
        self.seed = seed
        self.log_dir = Path(log_dir)
        self.csv_path: Path = self._build_csv_path()

        # Mirror DB handling on the CSV. log_alive_population_csv appends, and
        # db_handling="delete" only wipes the .db — so without this a re-run of
        # the same config would append a second experiment onto the first CSV.
        db_handling = ea_kwargs.get("db_handling") or config.db_handling
        if db_handling == "delete":
            csv_file = self.csv_path.with_suffix(".csv")
            if csv_file.exists():
                csv_file.unlink()

    def _build_csv_path(self) -> Path:
        """Build the CSV destination from experiment metadata."""
        if self.learning:
            prefix = "ldelta" if self.ldelta else "learning"
            name = (
                f"{prefix}{self.seed}_{self.pop_size}-{self.num_steps}"
                f"_{self.learning_epochs}_{self.tag}"
            )
        else:
            name = f"evo{self.seed}_{self.pop_size}-{self.num_steps}_{self.tag}"
        return self.log_dir / name

    def step(self) -> None:
        """Advance one generation, log alive population, report timing."""
        self.current_generation += 1
        self.fetch_population()
        start = time.time()
        for op in self.operations:
            self.population = op(self.population)
        log_alive_population_csv(
            self.population.alive,
            self.current_generation,
            self.csv_path,
        )
        self._commit()
        elapsed = time.time() - start
        self.console.log(
            f"Generation {self.current_generation}, in {elapsed:.2f}s",
        )

    def run(self) -> None:
        """Run the EA for ``num_steps`` generations with a progress bar."""
        # for _ in track(
        #     range(self.num_steps),
        #     description="Running EA:",
        #     console=self.console,
        # ):
        #     self.step()
        for _ in range(self.num_steps):
            self.step()
        self.console.rule("[green]EA Finished Running")
