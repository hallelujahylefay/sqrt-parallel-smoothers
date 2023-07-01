import jax.numpy as jnp


def transition_function(x, A, t):
    """ Deterministic transition function used in the inhomogeenous state space model
    Parameters
    ----------
    x: array_like
        The current state
    A: array_like
        transition matrix
    t: float
        time dependent transition function
    Returns
    -------
    out: array_like
        The transitioned state
    """
    return jnp.dot(A, x) * t


def observation_function(x, H, t):
    """
    Returns the observed angles as function of the state and the sensors locations
    Parameters
    ----------
    x: array_like
        The current state
    r: array_like
        The error param
    H: array_like
        observation matrix
    Returns
    -------
    y: array_like
        The observed data
    """
    return jnp.dot(H, x) * t ** 2