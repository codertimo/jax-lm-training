import argparse
import time
from typing import Any, Dict, List

import flax
import jax
import jax.numpy as jnp
import optax
import torch
from datasets import Dataset
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2LMHeadModel, GPT2Config

import wandb

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--model-config-path", type=str, default="resource/distil_gpt2_config.json")
parser.add_argument("--train-dataset-paths", type=str, default="dataset/wikitext.train**")
parser.add_argument("--train-batch-size", type=int, default=16)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--max-sequence-length", type=int, default=256)
parser.add_argument("--num-epochs", type=int, default=5)
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--weight-decay-rate", type=float, default=0.01)
parser.add_argument("--adamw-beta1", type=float, default=0.9)
parser.add_argument("--adamw-beta2", type=float, default=0.98)
parser.add_argument("--adamw-eps", type=float, default=1e-8)
parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
parser.add_argument("--wandb-username", default="codertimo")
parser.add_argument("--wandb-project", default="jax-lm-training")
# fmt: on


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


def main(args: argparse.Namespace):
    wandb.init(project=args.wandb_project, entity=args.wandb_username)
    wandb.config = dict(vars(args))

    train_dataset = Dataset.from_parquet(args.train_dataset_paths)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        drop_last=True,
        collate_fn=batch_collate_fn,
    )

    model_config = GPT2Config.from_json_file(args.model_config_path)
    model = FlaxGPT2LMHeadModel(
        model_config,
        input_shape=(args.train_batch_size, args.max_sequence_length),
        seed=0,
        dtype=jnp.dtype(args.dtype),
    )

    num_train_steps = len(dataloader) * args.num_epochs

    rng = jax.random.PRNGKey(args.random_seed)
    rng, dropout_rng = jax.random.split(rng)

    linear_decay_lr_schedule_fn = optax.linear_schedule(
        init_value=args.learning_rate, end_value=0, transition_steps=num_train_steps
    )
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=args.adamw_beta1,
        b2=args.adamw_beta2,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay_rate,
        mask=decay_mask_fn,
    )
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, num=2)

        def loss_fn(params):
            labels = batch.pop("labels")
            pred_logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            return optax.softmax_cross_entropy(pred_logits, onehot(labels, pred_logits.shape[-1])).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)
        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_dropout_rng

    parallel_train_step = jax.pmap(train_step, "batch")
    state = flax.jax_utils.replicate(state)

    train_metrics_stack = []
    last_timestamp = time.time()

    for epoch in range(args.num_epochs):
        dropout_rngs = jax.random.split(rng, num=jax.local_device_count())

        # TODO: device prefetch
        for i, batch in enumerate(dataloader):
            state, train_metric, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
            train_metrics_stack.append(train_metric)
            current_train_step = len(dataloader) * epoch + i

            if current_train_step > 0 and current_train_step % 100 == 0:
                train_metrics = get_metrics(train_metrics_stack)
                train_metrics = unreplicate(train_metrics)
                train_metrics = jax.tree_map(jnp.mean, train_metrics)
                loss = train_metrics["loss"]
                ppl = jnp.exp(loss)

                duration = int(time.time() - last_timestamp)
                eta = (num_train_steps - current_train_step) * duration // 50
                hour = eta // 3600
                minute = (eta % 3600) // 60
                second = eta % 60

                print(
                    f"epoch: {epoch} step: {current_train_step}/{num_train_steps} loss: {loss:.4f} ppl: {ppl:.2f} "
                    f"ETA {hour:02}:{minute:02}:{second:02}"
                )
                wandb.log({"loss": loss, "ppl": ppl, "epoch": epoch}, step=current_train_step)
                train_metrics_stack.clear()
                last_timestamp = time.time()


if __name__ == "__main__":
    main(parser.parse_args())
