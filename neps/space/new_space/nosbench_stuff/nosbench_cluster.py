from nosbench.utils import prune_program
from nosbench.program import Program
from neps.space.new_space.nosbench_spaces import Nosbench_space, Nosbench_space_int
import neps.space.new_space.space as space
import math
import nosbench
from functools import partial
import neps
import neps.space.new_space.bracket_optimizer as new_bracket_optimizer
import torch
import pprint

MAX_PROGRAM_LENGTH = 16
MAX_EPOCHS_PER_CONFIG = 20
AVAILABLE_VARIABLE_SLOTS = 11


def evaluate_pipeline(program: Program, epochs: int = 1, **_) -> float:
    # benchmark = nosbench.NOSBench()
    benchmark = nosbench.ToyBenchmark()
    prune_program(program)
    objective_to_minimize = benchmark.query(program, epochs)
    assert isinstance(objective_to_minimize, float)
    objective_to_minimize = (
        torch.inf if math.isnan(objective_to_minimize) else objective_to_minimize
    )
    return objective_to_minimize


optimizers_dict = {
    "RS": (
        space.RandomSearch,
        "new__RandomSearch",
    ),
    "CRS": (
        space.ComplexRandomSearch,
        "new__ComplexRandomSearch",
    ),
    "PB+SH": (
        partial(new_bracket_optimizer.priorband, base="successive_halving"),
        "new__priorband+successive_halving",
    ),
    "PB+ASHA": (
        partial(new_bracket_optimizer.priorband, base="asha"),
        "new__priorband+asha",
    ),
    "PB+ASHB": (
        partial(new_bracket_optimizer.priorband, base="async_hb"),
        "new__priorband+async_hb",
    ),
    "PB+HB": (
        new_bracket_optimizer.priorband,
        "new__priorband+hyperband",
    ),
}


def nosbench_neps_demo(
    optimizer,
    optimizer_name,
    max_evaluations_total=100,
    nosbench_dict={
        "max_program_length": MAX_PROGRAM_LENGTH,
        "max_epochs_per_config": MAX_EPOCHS_PER_CONFIG,
        "available_variable_slots": AVAILABLE_VARIABLE_SLOTS,
    },
):
    optimizer.__name__ = optimizer_name  # Needed by NEPS later.

    pipeline_space = Nosbench_space(
        max_program_length=nosbench_dict["max_program_length"],
        max_epochs_per_config=nosbench_dict["max_epochs_per_config"],
        available_variable_slots=nosbench_dict["available_variable_slots"],
    )
    pipeline_space = Nosbench_space_int(
        max_program_length=MAX_PROGRAM_LENGTH,
        max_epochs_per_config=MAX_EPOCHS_PER_CONFIG,
        available_variable_slots=AVAILABLE_VARIABLE_SLOTS,
    )
    root_directory = f"results/nosbench_{optimizer.__name__}"

    print(f"Running for root directory: {root_directory}")
    print(f"Using optimizer: {optimizer_name}")
    print(f"For {max_evaluations_total} evaluations in total.")
    pprint.pprint(nosbench_dict)

    neps.run(
        evaluate_pipeline=space.adjust_evaluation_pipeline_for_new_space(
            evaluate_pipeline,
            pipeline_space,
        ),
        pipeline_space=pipeline_space,
        optimizer=optimizer,
        root_directory=root_directory,
        post_run_summary=True,
        max_evaluations_total=max_evaluations_total,
        overwrite_working_directory=True,
    )

    neps.status(root_directory, print_summary=True)


if __name__ == "__main__":
    neps_dict = {
        "max_evaluations_total": 10,
        "optimizer": "RS",
    }
    nosbench_dict = {
        "max_program_length": MAX_PROGRAM_LENGTH,
        "max_epochs_per_config": MAX_EPOCHS_PER_CONFIG,
        "available_variable_slots": AVAILABLE_VARIABLE_SLOTS,
    }
    nosbench_neps_demo(
        *optimizers_dict[neps_dict["optimizer"]],
        max_evaluations_total=neps_dict["max_evaluations_total"],
    )
