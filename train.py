from absl import logging

# from flax import linen as nn
from flax import nnx

# from flax.metrics import tensorboard
# from flax.training import train_state
# import jax

from jax import random

# import jax.numpy as jnp
# import ml_collections
import numpy as np

# import optax
from tqdm.notebook import tqdm  # progress bar


# Train for a single epoch
def _train_epoch(
    model, loss_fn, optimizer, xs_train, ys_train, xs_test, ys_test, batch_size, rng
):
    train_ds_size = len(xs_train)
    steps_per_epoch = train_ds_size // batch_size

    def train_step(model, optimizer, loss_fn, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(
            model, x, y
        )  # get loss and gradients using JAX autodiff
        # update parameters of the model according to the optimizer
        optimizer.update(grads)
        return loss

    perms = random.permutation(
        rng, len(xs_train)
    )  # shuffle dataset (permutation of samples order)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []

    for perm in perms:
        batch_xs = xs_train[perm, :]
        batch_ys = ys_train[perm, :]
        loss = train_step(model, optimizer, loss_fn, batch_xs, batch_ys)
        epoch_loss.append(loss)  # store training loss for the current batch

    train_loss = np.mean(epoch_loss)
    test_loss = loss_fn(model, xs_test, ys_test)
    return model, train_loss, test_loss


def train(
    model,
    optimizer,
    loss_fn,
    xs_train,
    xs_test,
    ys_train,
    ys_test,
    batch_size,
    epochs,
    log_period_epoch=1,
    show_progress=True,
):

    train_loss_history = []
    test_loss_history = []

    for epoch in tqdm(range(1, epochs + 1), disable=not show_progress):
        model, train_loss, test_loss = _train_epoch(
            model,
            loss_fn,
            optimizer,
            xs_train,
            ys_train,
            xs_test,
            ys_test,
            batch_size,
            random.key(1),
        )

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if epoch == 1 or epoch % log_period_epoch == 0:
            logging.info(
                "epoch:% 3d, train_loss: %.4f, test_loss: %.4f,"
                % (
                    epoch,
                    train_loss,
                    test_loss,
                )
            )
    return train_loss_history, test_loss_history


# def create_train_state(rng, config):
#     """Creates initial `TrainState`."""
#     cnn = CNN()
#     params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
#     tx = optax.sgd(config.learning_rate, config.momentum)
#     return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


# def train_and_evaluate(
#     config: ml_collections.ConfigDict, workdir: str
# ) -> train_state.TrainState:
#     """Execute model training and evaluation loop.

#     Args:
#       config: Hyperparameter configuration for training and evaluation.
#       workdir: Directory where the tensorboard summaries are written to.

#     Returns:
#       The train state (which includes the `.params`).
#     """
#     train_ds, test_ds = get_datasets()
#     rng = jax.random.key(0)

#     summary_writer = tensorboard.SummaryWriter(workdir)
#     summary_writer.hparams(dict(config))

#     rng, init_rng = jax.random.split(rng)
#     state = create_train_state(init_rng, config)

#     for epoch in range(1, config.num_epochs + 1):
#         rng, input_rng = jax.random.split(rng)
#         state, train_loss, train_accuracy = train_epoch(
#             state, train_ds, config.batch_size, input_rng
#         )
#         _, test_loss, test_accuracy = apply_model(
#             state, test_ds["image"], test_ds["label"]
#         )

#         logging.info(
#             "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,"
#             " test_accuracy: %.2f"
#             % (
#                 epoch,
#                 train_loss,
#                 train_accuracy * 100,
#                 test_loss,
#                 test_accuracy * 100,
#             )
#         )

#         summary_writer.scalar("train_loss", train_loss, epoch)
#         summary_writer.scalar("train_accuracy", train_accuracy, epoch)
#         summary_writer.scalar("test_loss", test_loss, epoch)
#         summary_writer.scalar("test_accuracy", test_accuracy, epoch)

#     summary_writer.flush()
#     return state
