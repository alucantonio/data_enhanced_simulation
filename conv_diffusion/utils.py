import jax
import jax.numpy as jnp


def compute_u_v_f(k):
    # exact solution
    def u_fun(x, y):
        return jnp.sin(5 * jnp.pi * x) * jnp.sin(5 * jnp.pi * y)
        # return jnp.exp(-5.0 * (x + y - 1.0))

    # gradient ∇u
    def grad_u(xy):
        return jax.grad(lambda z: u_fun(z[0], z[1]))(xy)

    # Laplacian Δu
    def laplace_u(xy):
        H = jax.hessian(lambda z: u_fun(z[0], z[1]))(xy)
        return jnp.trace(H)  # u_xx + u_yy

    # velocity field v(x,y)
    def v_fun(x, y):
        # divergence-free example:
        return jnp.array([y - 0.5, -(x - 0.5)])
        # compressible example (div v = 0):
        # return jnp.array([5.0, 5.0])

    # divergence of v
    def div_v(x, y):
        def vx(xy):
            return v_fun(xy[0], xy[1])[0]

        def vy(xy):
            return v_fun(xy[0], xy[1])[1]

        dvx_dx = jax.grad(lambda z: vx(jnp.array([z, y])))(x)
        dvy_dy = jax.grad(lambda z: vy(jnp.array([x, z])))(y)
        return dvx_dx + dvy_dy

    # forcing term f(x,y)
    def f_fun(x, y):
        xy = jnp.array([x, y])
        grad_val = grad_u(xy)
        laplace_val = laplace_u(xy)
        v_val = v_fun(x, y)
        div_v_val = div_v(x, y)
        return -k * laplace_val + jnp.dot(v_val, grad_val) + div_v_val * u_fun(x, y)

    # vectorize over arrays of nodes
    # f_fun_vec = jax.vmap(jax.vmap(f_fun, (0, None)), (None, 0))
    u_fun_vec = jax.vmap(jax.vmap(u_fun, (0, None)), (None, 0))

    return u_fun_vec, v_fun, f_fun
