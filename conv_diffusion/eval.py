import os
import jax.numpy as jnp


from jaxpi.utils import restore_checkpoint

import ml_collections


from matplotlib import pyplot as plt
from matplotlib import tri
import matplotlib.colors as mcolors

from utils import compute_u_v_f
from models import ConvectionDiffusionPINN, ConvectionDiffusionPINNEvaluator


def evaluate(config: ml_collections.ConfigDict, workdir: str):
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

    # Restore model and compute errors/predictions
    model = ConvectionDiffusionPINN(config, v_fun, f_fun, k, x_star, y_star, bvalues)
    ckpt_dir = "ckpt"
    ckpt_path = os.path.join(workdir, config.wandb.name, ckpt_dir)
    model.state = restore_checkpoint(model.state, ckpt_path)
    u_pred = model.u_pred_fn_vec(model.state.params, x_star, y_star)
    print("L2 error:", model.compute_l2_error(model.state.params, u_true))

    # -- PLOTS --
    x_mesh, y_mesh = jnp.meshgrid(x_star, y_star)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = jnp.min(u_true), jnp.max(u_true)
    norm, cmap = mcolors.Normalize(vmin=vmin, vmax=vmax), "RdBu"

    titles = ["Data", "PINN", "Data - PINN"]
    fields = [u_true, u_pred, u_true - u_pred]

    u_plot = None
    for ax, field, title in zip(axes, fields, titles):
        u_plot = ax.contourf(x_mesh, y_mesh, field, cmap=cmap, levels=100)
        plt.colorbar(u_plot, ax=ax, label=r"$u$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("conv_diffusion_contour.png", dpi=300)

    fig = plt.figure(figsize=(20, 6))

    for i, (field, title) in enumerate(zip(fields, titles), 1):
        ax = fig.add_subplot(1, len(fields), i, projection="3d")
        surf = ax.plot_surface(
            x_mesh,
            y_mesh,
            field,
            cmap="viridis",
            linewidth=0.2,
            antialiased=True,
        )
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        # ax.set_zlabel("u", labelpad=8)
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

    plt.tight_layout()
    plt.savefig("conv_diffusion_3d.png", dpi=300, bbox_inches="tight")
