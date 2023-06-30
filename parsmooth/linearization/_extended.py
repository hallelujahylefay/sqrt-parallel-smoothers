from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, ConditionalMomentsModel, MVNSqrt, are_inputs_compatible, MVNStandard


def linearize(model: Union[FunctionalModel, ConditionalMomentsModel], x: Union[MVNSqrt, MVNStandard], param=None):
    """
    Extended linearization for a non-linear function f(x, q). If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.

    Parameters
    ----------
    model: Union[FunctionalModel, ConditionalMomentsModel]
        The function to be called on x and q
    x: Union[MVNSqrt, MVNStandard]
        x-coordinate state at which to linearize f

    Returns
    -------
    F_x, F_q, res: jnp.ndarray
        The linearization parameters
    chol_q or cov_q: jnp.ndarray
        Either the cholesky or the full-rank modified covariance matrix.
    """
    if isinstance(model, FunctionalModel):
        f, q = model
        are_inputs_compatible(x, q)

        m_x, _ = x
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_callable(f, m_x, *q, param)
        return _standard_linearize_callable(f, m_x, *q, param)

    else:
        if isinstance(x, MVNSqrt):
            return _sqrt_linearize_conditional(model.conditional_mean, model.conditional_covariance_or_cholesky, x, param)
        return _standard_linearize_conditional(model.conditional_mean, model.conditional_covariance_or_cholesky, x, param)


def _standard_linearize_conditional(c_m, c_cov, x, param=None):
    m, p = x
    y = (m,) if param is None else (m, *param)
    F = jax.jacfwd(c_m, 0)(*y)
    b = c_m(*y) - F @ m
    Cov = c_cov(*y)
    return F, Cov, b


def _sqrt_linearize_conditional(c_m, c_chol, x, param=None):
    m, _ = x
    y = (m,) if param is None else (m, *param)
    F = jax.jacfwd(c_m, 0)(*y)
    b = c_m(*y) - F @ m
    Chol = c_chol(*y)
    return F, Chol, b


def _linearize_callable_common(f, x, param=None) -> Tuple[Any, Any]:
    y = (x,) if param is None else (x, *param)
    return f(*y), jax.jacfwd(f, 0)(*y)


def _standard_linearize_callable(f, x, m_q, cov_q, param=None):
    y = (x,) if param is None else (x, *param)
    res, F_x = _linearize_callable_common(f, *y)
    return F_x, cov_q, res - F_x @ x + m_q


def _sqrt_linearize_callable(f, x, m_q, chol_q, param=None):
    y = (x,) if param is None else (x, *param)
    res, F_x = _linearize_callable_common(f, *y)
    return F_x, chol_q, res - F_x @ x + m_q
