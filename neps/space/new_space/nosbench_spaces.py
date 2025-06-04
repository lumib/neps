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
        max_program_length=MAX_PROGRAM_LENGTH,
        max_epochs_per_config=MAX_EPOCHS_PER_CONFIG,
        available_variable_slots=AVAILABLE_VARIABLE_SLOTS,
        epoch_fidelity=True,
        **_,
    ):
        self.max_program_length = max_program_length
        self.max_epochs_per_config = max_epochs_per_config
        self.available_variable_slots = available_variable_slots
        self.epoch_fidelity = epoch_fidelity

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
        max_program_length=MAX_PROGRAM_LENGTH,
        max_epochs_per_config=MAX_EPOCHS_PER_CONFIG,
        available_variable_slots=AVAILABLE_VARIABLE_SLOTS,
        epoch_fidelity=True,
        **_,
    ):
        self.max_program_length = max_program_length
        self.max_epochs_per_config = max_epochs_per_config
        self.available_variable_slots = available_variable_slots
        self.epoch_fidelity = epoch_fidelity

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

adamw_samplings = ({
    'Resolvable.program.args.resampled_categorical::categorical__20': 13,
    'Resolvable.program.args[0].resampled_operation.args.resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[0].resampled_operation.args[0].resampled_categorical::categorical__16': 0,
    'Resolvable.program.args[0].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[0].resampled_operation.args[1].resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[0].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 11,

    'Resolvable.program.args[1].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[1].resampled_operation.args[0].resampled_categorical::categorical__8': 4,
    'Resolvable.program.args[1].resampled_operation.args[1].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[1].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 3,
    'Resolvable.program.args[1].resampled_operation.args[2].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[1].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 5,
    'Resolvable.program.args[1].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 9,

    'Resolvable.program.args[2].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[2].resampled_operation.args[0].resampled_categorical::categorical__8': 4,
    'Resolvable.program.args[2].resampled_operation.args[1].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[2].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 3,
    'Resolvable.program.args[2].resampled_operation.args[2].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[2].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 7,
    'Resolvable.program.args[2].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 10,

    'Resolvable.program.args[3].resampled_operation.args.resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[3].resampled_operation.args[0].resampled_categorical::categorical__2': 0,
    'Resolvable.program.args[3].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 12,
    'Resolvable.program.args[3].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[3].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[3].resampled_operation.args[2].resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[3].resampled_operation.args[3].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 9,
    'Resolvable.program.args[3].resampled_operation.args[3].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[3].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 12,

    'Resolvable.program.args[4].resampled_operation.args.resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[4].resampled_operation.args[0].resampled_categorical::categorical__2': 0,
    'Resolvable.program.args[4].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 13,
    'Resolvable.program.args[4].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[4].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 11,
    'Resolvable.program.args[4].resampled_operation.args[2].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[4].resampled_operation.args[3].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 10,
    'Resolvable.program.args[4].resampled_operation.args[3].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[4].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 13,

    'Resolvable.program.args[5].resampled_operation.args.resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[5].resampled_operation.args[0].resampled_categorical::categorical__2': 1,
    'Resolvable.program.args[5].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 12,
    'Resolvable.program.args[5].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[5].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 9,
    'Resolvable.program.args[5].resampled_operation.args[2].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[5].resampled_operation.args[3].resampled_categorical.sampled_value.resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[5].resampled_operation.args[3].resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[5].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 14,

    'Resolvable.program.args[6].resampled_operation.args.resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[6].resampled_operation.args[0].resampled_categorical::categorical__2': 1,
    'Resolvable.program.args[6].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 13,
    'Resolvable.program.args[6].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[6].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 10,
    'Resolvable.program.args[6].resampled_operation.args[2].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[6].resampled_operation.args[3].resampled_categorical.sampled_value.resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[6].resampled_operation.args[3].resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[6].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 15,

    'Resolvable.program.args[7].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[7].resampled_operation.args[0].resampled_categorical::categorical__8': 2,
    'Resolvable.program.args[7].resampled_operation.args[1].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[7].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 6,
    'Resolvable.program.args[7].resampled_operation.args[2].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[7].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 8,
    'Resolvable.program.args[7].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 16,

    'Resolvable.program.args[8].resampled_operation.args.resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[8].resampled_operation.args[0].resampled_categorical::categorical__16': 4,
    'Resolvable.program.args[8].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[8].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 15,
    'Resolvable.program.args[8].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 17,

    'Resolvable.program.args[9].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[9].resampled_operation.args[0].resampled_categorical::categorical__8': 3,
    'Resolvable.program.args[9].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[9].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 17,
    'Resolvable.program.args[9].resampled_operation.args[2].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[9].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 16,
    'Resolvable.program.args[9].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 17,

    'Resolvable.program.args[10].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[10].resampled_operation.args[0].resampled_categorical::categorical__8': 1,
    'Resolvable.program.args[10].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[10].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 14,
    'Resolvable.program.args[10].resampled_operation.args[2].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[10].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 17,
    'Resolvable.program.args[10].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 19,

    'Resolvable.program.args[11].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[11].resampled_operation.args[0].resampled_categorical::categorical__8': 2,
    'Resolvable.program.args[11].resampled_operation.args[1].resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[11].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_categorical::categorical__3': 0,
    'Resolvable.program.args[11].resampled_operation.args[2].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[11].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 6,
    'Resolvable.program.args[11].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 18,

    'Resolvable.program.args[12].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[12].resampled_operation.args[0].resampled_categorical::categorical__8': 3,
    'Resolvable.program.args[12].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[12].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 19,
    'Resolvable.program.args[12].resampled_operation.args[2].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[12].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 18,
    'Resolvable.program.args[12].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 19,

    'Resolvable.program.args[13].resampled_operation.args.resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[13].resampled_operation.args[0].resampled_categorical::categorical__8': 2,
    'Resolvable.program.args[13].resampled_operation.args[1].resampled_categorical::categorical__3': 2,
    'Resolvable.program.args[13].resampled_operation.args[1].resampled_categorical.sampled_value.resampled_integer::integer__9_20_False': 19,
    'Resolvable.program.args[13].resampled_operation.args[2].resampled_categorical::categorical__3': 1,
    'Resolvable.program.args[13].resampled_operation.args[2].resampled_categorical.sampled_value.resampled_integer::integer__3_8_False': 7,
    'Resolvable.program.args[13].resampled_operation.kwargs{store}.resampled_integer::integer__9_20_False': 19,
},
{'epochs': 20})
