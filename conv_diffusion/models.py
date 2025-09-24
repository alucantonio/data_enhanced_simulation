from jax import jit, grad, vmap, jacfwd
import jax.numpy as jnp


from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator


from functools import partial


class ConvectionDiffusionPINN(ForwardBVP):
    def __init__(self, config, v_fun, f_fun, k, x_star, y_star, bvalues):
        super().__init__(config)

        self.v_fun = v_fun
        self.f_fun = f_fun
        self.k = k
        self.x_star = x_star
        self.y_star = y_star
        self.bvalues = bvalues

        # Predictions over a grid: vectorizing both in x and y
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.u_pred_fn_vec = vmap(vmap(self.u_net, (None, 0, None)), (None, None, 0))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))
        self.res_term = vmap(self.r_net, (None, 0, 0))

        # derivatives definition
        self.u_x = grad(self.u_net, argnums=1)
        self.u_xx = grad(grad(self.u_net, argnums=1), argnums=1)
        self.u_y = grad(self.u_net, argnums=2)
        self.u_yy = grad(grad(self.u_net, argnums=2), argnums=2)
        self.jac_v = jacfwd(self.v_fun, argnums=(0, 1))

    def u_net(self, params, x, y):
        z = jnp.stack([x, y])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, x, y):
        """Compute the residual using automatic differentiation"""
        u = self.u_net(params, x, y)
        v = self.v_fun(x, y)
        jac_v = self.jac_v(x, y)

        # compute diffusion term
        laplacian = self.u_xx(params, x, y) + self.u_yy(params, x, y)
        diff_term = self.k * laplacian

        # compute convective term
        conv_term = (
            u * (jac_v[0][0] + jac_v[1][1])
            + v[0] * self.u_x(params, x, y)
            + v[1] * self.u_y(params, x, y)
        )

        residual = diff_term - conv_term + self.f_fun(x, y)
        return residual

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # compute bnd values
        u_pred_x = self.u_pred_fn_vec(params, self.x_star, jnp.array([0, 1]))
        u_pred_y = self.u_pred_fn_vec(params, jnp.array([0, 1]), self.y_star)

        # Boundary condition loss
        bnd_loss = jnp.mean((u_pred_x - self.bvalues[0]) ** 2) + jnp.mean(
            (u_pred_y - self.bvalues[1]) ** 2
        )

        # Residual loss
        r_pred = self.res_term(params, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {
            "res": res_loss,
            "bnd": bnd_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_ref):
        u_pred = self.u_pred_fn_vec(params, self.x_star, self.y_star)
        error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        return error


class ConvectionDiffusionPINNEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        return self.log_dict
