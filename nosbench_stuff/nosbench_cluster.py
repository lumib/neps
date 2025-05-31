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
import argparse

MAX_PROGRAM_LENGTH = 16
MAX_EPOCHS_PER_CONFIG = 20
AVAILABLE_VARIABLE_SLOTS = 11


def evaluate_pipeline(program: Program, epochs: int = MAX_EPOCHS_PER_CONFIG, benchmark= nosbench.ToyBenchmark(), **_) -> float:
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
    rep_suffix="",
    nosbench_dict={
        "max_program_length": MAX_PROGRAM_LENGTH,
        "max_epochs_per_config": MAX_EPOCHS_PER_CONFIG,
        "available_variable_slots": AVAILABLE_VARIABLE_SLOTS,
        "benchmark": nosbench.ToyBenchmark(),
        "pipeline_space": Nosbench_space(nosbench_dict={
            "max_program_length": MAX_PROGRAM_LENGTH,
            "max_epochs_per_config": MAX_EPOCHS_PER_CONFIG,
            "available_variable_slots": AVAILABLE_VARIABLE_SLOTS,
            }
        ),
    },
):
    optimizer.__name__ = optimizer_name  # Needed by NEPS later.

    

    root_directory = f"results/nosbench_{optimizer.__name__}{'_'+rep_suffix if rep_suffix else ''}"

    print(f"Running for root directory: {root_directory}")
    print(f"Using optimizer: {optimizer_name}")
    print(f"For {max_evaluations_total} evaluations in total.")
    pprint.pprint(nosbench_dict)

    neps.run(
        evaluate_pipeline=space.adjust_evaluation_pipeline_for_new_space(
            partial(evaluate_pipeline, benchmark=nosbench_dict["benchmark"], epochs=nosbench_dict["max_epochs_per_config"]),
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
    parser = argparse.ArgumentParser(description="Run NEPS with Nosbench on a cluster.")
    parser.add_argument(
        "--optimizer","-o",
        type=str,
        choices=list(optimizers_dict.keys()),
        default="PB+ASHB",
        help="Optimizer to use for the NEPS run. Available options: " + ", ".join(optimizers_dict.keys()),
    )
    parser.add_argument(
        "--max_evaluations_total","-ev",
        type=int,
        default=100,
        help="Total number of evaluations to run.",
    )
    parser.add_argument(
        "--rep_suffix", "-rs",
        type=str,
        default="",
        help="Suffix to append to the results directory.",
    )
    parser.add_argument(
        "--program_length", "-pl",
        type=int,
        default=MAX_PROGRAM_LENGTH,
        help="Maximum program length for the Nosbench space.",
    )
    parser.add_argument(
        "--epochs", "-ep",
        type=int,
        default=MAX_EPOCHS_PER_CONFIG,
        help="Maximum epochs per configuration for the Nosbench space.",
    )
    parser.add_argument(
        "--available_variable_slots",
        type=int,
        default=AVAILABLE_VARIABLE_SLOTS,
        help="Number of available variable slots for the Nosbench space.",
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        default="ToyBenchmark",
        choices=["ToyBenchmark", "NosBench"],
        help="Benchmark to use for the Nosbench space. Available options: ToyBenchmark, NosBench.",
    )
    parser.add_argument(
        "--pipeline_space_int","-psi",
        action="store_true",
        help="Use the integer version of the Nosbench space.",
    )
    args = parser.parse_args()

    neps_dict = {
        "max_evaluations_total": args.max_evaluations_total,
        "optimizer": args.optimizer,
        "directory_suffix": args.rep_suffix,
    }
    nosbench_dict = {
        "max_program_length": args.program_length,
        "max_epochs_per_config": args.epochs,
        "available_variable_slots": args.available_variable_slots,
        "benchmark": nosbench.ToyBenchmark() if args.benchmark == "ToyBenchmark" else nosbench.NosBench() if args.benchmark == "NosBench" else nosbench.ToyBenchmark(),
    }
    pipeline_space = Nosbench_space(nosbench_dict=nosbench_dict)
    pipeline_space_int = Nosbench_space_int(nosbench_dict=nosbench_dict)
    nosbench_dict["pipeline_space"] = pipeline_space if not args.pipeline_space_int else pipeline_space_int

    nosbench_neps_demo(
        *optimizers_dict[neps_dict["optimizer"]],
        max_evaluations_total=neps_dict["max_evaluations_total"],
        rep_suffix=neps_dict["directory_suffix"],
        nosbench_dict=nosbench_dict,
    )
