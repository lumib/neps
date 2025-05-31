from nosbench.program import Program, Instruction, Pointer
from nosbench.function import Function, interpolate, bias_correct, clip, size
import torch

from neps.space.new_space import space

MAX_PROGRAM_LENGTH = 20  # As per the Nosbench paper
MAX_EPOCHS_PER_CONFIG = 20  # As per the Nosbench paper
AVAILABLE_VARIABLE_SLOTS = 11  # As per the Nosbench paper


class Nosbench_space(space.Pipeline):
    def __init__(
        self,
        nosbench_dict={
            "max_program_length": MAX_PROGRAM_LENGTH,
            "max_epochs_per_config": MAX_EPOCHS_PER_CONFIG,
            "available_variable_slots": AVAILABLE_VARIABLE_SLOTS,
            "epoch_fidelity": True,
        },
    ):
        self.max_program_length = nosbench_dict["max_program_length"]
        self.max_epochs_per_config = nosbench_dict["max_epochs_per_config"]
        self.available_variable_slots = nosbench_dict["available_variable_slots"]
        self.epoch_fidelity = nosbench_dict["epoch_fidelity"]

        self._UNARY_FUN = space.Categorical(
            choices=(
                torch.square,
                torch.exp,
                torch.log,
                torch.sign,
                torch.sqrt,
                torch.abs,
                torch.norm,
                torch.sin,
                torch.cos,
                torch.tan,
                torch.asin,
                torch.acos,
                torch.atan,
                torch.mean,
                torch.std,
                size,
            )
        )

        self._BINARY_FUN = space.Categorical(
            choices=(
                clip,
                torch.div,
                torch.mul,
                torch.add,
                torch.sub,
                torch.minimum,
                torch.maximum,
                torch.heaviside,
            )
        )

        self._TERNARY_FUN = space.Categorical(
            choices=(
                interpolate,
                bias_correct,
            )
        )

        self._CONST = space.Integer(3, 8)
        self._VAR = space.Integer(9, self.available_variable_slots + 9)

        self._PARAMS = space.Categorical(choices=(0, 1, 2))

        self._POINTER = space.Categorical(
            choices=(
                space.Resampled(self._PARAMS),
                space.Resampled(self._CONST),
                space.Resampled(self._VAR),
            ),
        )

        self._F_ARGS = space.Categorical(
            choices=(
                (
                    space.Resampled(self._UNARY_FUN),
                    space.Resampled(self._POINTER),
                ),
                (
                    space.Resampled(self._BINARY_FUN),
                    space.Resampled(self._POINTER),
                    space.Resampled(self._POINTER),
                ),
                (
                    space.Resampled(self._TERNARY_FUN),
                    space.Resampled(self._POINTER),
                    space.Resampled(self._POINTER),
                    space.Resampled(self._POINTER),
                ),
            ),
        )

        self._F = space.Operation(
            operator=self.function_writer,
            args=space.Resampled(self._F_ARGS),
            kwargs={"store": space.Resampled(self._VAR)},
        )

        self._P_ARGS = space.Categorical(
            choices=tuple([
                (space.Resampled(self._F),) * i
                for i in range(1, self.max_program_length + 1)
            ]),
        )

        self.program = space.Operation(
            operator=self.program_compiler,
            args=space.Resampled(self._P_ARGS),
        )
        if self.epoch_fidelity:
            self.epochs = space.Fidelity(space.Integer(1, self.max_epochs_per_config))

    @staticmethod
    def function_writer(operation, *args, store: int):
        return (
            Function(operation, len(args)),
            [Pointer(var) if isinstance(var, int) else Pointer(var()) for var in args],
            Pointer(store),
        )

    @staticmethod
    def program_compiler(*args):
        return Program([
            Instruction(instruction[0], inputs=instruction[1], output=instruction[2])
            for instruction in args
        ])


class Nosbench_space_int(space.Pipeline):
    def __init__(
        self,
        nosbench_dict={
            "max_program_length": MAX_PROGRAM_LENGTH,
            "max_epochs_per_config": MAX_EPOCHS_PER_CONFIG,
            "available_variable_slots": AVAILABLE_VARIABLE_SLOTS,
            "epoch_fidelity": True,
        },
    ):
        self.max_program_length = nosbench_dict["max_program_length"]
        self.max_epochs_per_config = nosbench_dict["max_epochs_per_config"]
        self.available_variable_slots = nosbench_dict["available_variable_slots"]
        self.epoch_fidelity = nosbench_dict["epoch_fidelity"]

        self._UNARY_FUN = space.Categorical(
            choices=(
                torch.square,
                torch.exp,
                torch.log,
                torch.sign,
                torch.sqrt,
                torch.abs,
                torch.norm,
                torch.sin,
                torch.cos,
                torch.tan,
                torch.asin,
                torch.acos,
                torch.atan,
                torch.mean,
                torch.std,
                size,
            )
        )

        self._BINARY_FUN = space.Categorical(
            choices=(
                clip,
                torch.div,
                torch.mul,
                torch.add,
                torch.sub,
                torch.minimum,
                torch.maximum,
                torch.heaviside,
            )
        )

        self._TERNARY_FUN = space.Categorical(
            choices=(
                interpolate,
                bias_correct,
            )
        )

        self._CONST = space.Integer(3, 8)
        self._VAR = space.Integer(9, self.available_variable_slots + 9)

        self._PARAMS = space.Categorical(choices=(0, 1, 2))

        self._POINTER = space.Categorical(
            choices=(
                space.Resampled(self._PARAMS),
                space.Resampled(self._CONST),
                space.Resampled(self._VAR),
            ),
        )

        self._F_ARGS = space.Categorical(
            choices=(
                (
                    space.Resampled(self._UNARY_FUN),
                    space.Resampled(self._POINTER),
                ),
                (
                    space.Resampled(self._BINARY_FUN),
                    space.Resampled(self._POINTER),
                    space.Resampled(self._POINTER),
                ),
                (
                    space.Resampled(self._TERNARY_FUN),
                    space.Resampled(self._POINTER),
                    space.Resampled(self._POINTER),
                    space.Resampled(self._POINTER),
                ),
            ),
        )

        self._F = space.Operation(
            operator=self.function_writer,
            args=space.Resampled(self._F_ARGS),
            kwargs={"store": space.Resampled(self._VAR)},
        )

        self.program = space.Operation(
            operator=self.program_compiler,
            args=(space.Resampled(self._F),) * self.max_program_length,
            kwargs={"n_lines": space.Integer(1, self.max_program_length)},
        )

        if self.epoch_fidelity:
            self.epochs = space.Fidelity(space.Integer(1, self.max_epochs_per_config))

    @staticmethod
    def function_writer(
        operation, *args, store: int
    ) -> tuple[Function, list[Pointer], Pointer]:
        return (
            Function(operation, len(args)),
            [Pointer(var) if isinstance(var, int) else Pointer(var()) for var in args],
            Pointer(store),
        )

    @staticmethod
    def program_compiler(*args, n_lines: int = 20) -> Program:
        return Program([
            Instruction(instruction[0], inputs=instruction[1], output=instruction[2])
            for instruction in args[:n_lines]
        ])
