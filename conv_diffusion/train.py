import os
import jax
from jax.tree_util import tree_map
import jax.numpy as jnp


from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint
from jaxpi.samplers import UniformSampler

import wandb
import ml_collections
import time

from utils import compute_u_v_f
from models import ConvectionDiffusionPINN, ConvectionDiffusionPINNEvaluator


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # -- DATASET --
    x_star = jnp.linspace(0.0, 1.0, 100)
    y_star = jnp.linspace(0.0, 1.0, 100)
    k = 0.0001

    u_fun, v_fun, f_fun = compute_u_v_f(k)

    # define u true and boundary conditions
    u_true = u_fun(x_star, y_star)
    b_values_x = u_fun(x_star, jnp.array([0, 1]))
    b_values_y = u_fun(jnp.array([0, 1]), y_star)
    bvalues = (b_values_x, b_values_y)

    # Initialize model
    model = ConvectionDiffusionPINN(config, v_fun, f_fun, k, x_star, y_star, bvalues)
    # Initialize evaluator
    evaluator = ConvectionDiffusionPINNEvaluator(config, model)

    # Define domain
    dom = jnp.array([[0.0, 1.0], [0.0, 1.0]])

    # Initialize residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))

                log_dict = evaluator(state, batch, u_true)
                wandb.log(log_dict, step)
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(
                    model.state, ckpt_path, keep=config.saving.num_keep_ckpts
                )

    return model
