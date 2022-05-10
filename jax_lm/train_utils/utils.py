from typing import Any, Dict, List

import jax.numpy as jnp
from flax.training.common_utils import shard
from flax.traverse_util import flatten_dict, unflatten_dict


def batch_collate_fn(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_dict = {key: [] for key in data_list[0].keys()}
    for data in data_list:
        for key, value in data.items():
            batch_dict[key].append(value)
    return shard({key: jnp.array(value) for key, value in batch_dict.items()})


def decay_mask_fn(params):
    flat_params = flatten_dict(params)
    flat_mask = {
        path: (path[-1] != "bias" and path[-2:] not in [("ln_1", "scale"), ("ln_2", "scale"), ("ln_f", "scale")])
        for path in flat_params
    }
    return unflatten_dict(flat_mask)
