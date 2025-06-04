from nosbench.utils import prune_program
from nosbench.program import Program
from nosbench.optimizers import AdamW
from neps.space.new_space.nosbench_spaces import Nosbench_space, Nosbench_space_int
import neps.space.new_space.space as space
import nosbench
import pprint

import yaml

# experiment_name = "nosbench_new__priorband+async_hb_4_GPU_100k_wo_Overwrite"
experiment_name = "2_5k_w_warmstart"
config_id = "1_2"
MAX_PROGRAM_LENGTH = 20
MAX_EPOCHS_PER_CONFIG = 20
AVAILABLE_VARIABLE_SLOTS = 11
nosbench_space = Nosbench_space(max_program_length=20, max_epochs_per_config=20, available_variable_slots=11)

with open("./results/"+experiment_name+"/configs/config_"+config_id+"/config.yaml") as stream:
    config=yaml.safe_load(stream)
    # pprint.pprint(config)

# For each key, remove the 'SAMPLING__' prefix, except for 'epochs'
for key in list(config.keys()):
    if key.startswith('SAMPLING__'):
        new_key = key[len('SAMPLING__'):]
        config[new_key] = config.pop(key)

# Collect the ENVIRONMENT__ values in a separate dictionary without the prefix
environment_values = {}
for key in list(config.keys()):
    if key.startswith('ENVIRONMENT__'):
        new_key = key[len('ENVIRONMENT__'):]
        environment_values[new_key] = config.pop(key)

resolved_pipeline, resolution_context = space.resolve(nosbench_space, space.OnlyPredefinedValuesSampler(config), environment_values=environment_values)
program=space.convert_operation_to_callable(resolved_pipeline.program)
pprint.pprint(program)
print(program==AdamW)
prune_program(program)
# print(nosbench.NOSBench(path="test_cache").query(program, epoch=environment_values['epochs']))
# print()
# pprint.pprint(program)
# print(nosbench.NOSBench(path="test_cache").query(program, epoch=environment_values['epochs']))
