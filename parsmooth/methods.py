from typing import Callable, Optional, Union, Tuple

from jax import numpy as jnp

from parsmooth._base import MVNStandard, FunctionalModel, MVNSqrt, ConditionalMomentsModel
from parsmooth._pathwise_sampler import _par_sampling, _seq_sampling
from parsmooth._utils import fixed_point
from parsmooth.parallel._filtering import filtering as par_filtering
from parsmooth.parallel._smoothing import smoothing as par_smoothing
from parsmooth.sequential._filtering import filtering as seq_filtering
from parsmooth.sequential._smoothing import smoothing as seq_smoothing


def filtering(observations: jnp.ndarray,
              x0: Union[MVNSqrt, MVNStandard],
              transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              observation_model: Union[FunctionalModel, ConditionalMomentsModel],
              linearization_method: Callable,
              nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
              parallel: bool = True,
              return_loglikelihood: bool = False,
              params_transition: Tuple = None,
              params_observation: Tuple = None):
    if parallel:
        return par_filtering(observations, x0, transition_model, observation_model, linearization_method,
                             nominal_trajectory, return_loglikelihood, params_transition, params_observation)
    return seq_filtering(observations, x0, transition_model, observation_model, linearization_method,
                         nominal_trajectory, return_loglikelihood, params_transition, params_observation)


def smoothing(transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              filter_trajectory: Union[MVNSqrt, MVNStandard],
              linearization_method: Callable, nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
              parallel: bool = True,
              params_transition: Tuple = None):
    if parallel:
        return par_smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory, params_transition)
    return seq_smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory, params_transition)


def filter_smoother(observations: jnp.ndarray,
                    x0: Union[MVNSqrt, MVNStandard],
                    transition_model: Union[FunctionalModel, ConditionalMomentsModel],
                    observation_model: Union[FunctionalModel, ConditionalMomentsModel],
                    linearization_method: Callable,
                    nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
                    parallel: bool = True,
                    params_transition: Tuple = None,
                    params_observation: Tuple = None):
    filter_trajectory = filtering(observations, x0, transition_model, observation_model, linearization_method,
                                  nominal_trajectory, parallel, params_transition, params_observation)
    return smoothing(transition_model, filter_trajectory, linearization_method, nominal_trajectory, parallel)


def _default_criterion(_i, nominal_traj_prev, curr_nominal_traj):
    return jnp.mean((nominal_traj_prev.mean - curr_nominal_traj.mean) ** 2) > 1e-6


def iterated_smoothing(observations: jnp.ndarray,
                       x0: Union[MVNSqrt, MVNStandard],
                       transition_model: Union[FunctionalModel, ConditionalMomentsModel],
                       observation_model: Union[FunctionalModel, ConditionalMomentsModel],
                       linearization_method: Callable,
                       init_nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
                       parallel: bool = True,
                       criterion: Callable = _default_criterion,
                       return_loglikelihood: bool = False,
                       params_transition: Tuple = None,
                       params_observation: Tuple = None
                       ):
    if init_nominal_trajectory is None:
        init_nominal_trajectory = filter_smoother(observations, x0, transition_model, observation_model,
                                                  linearization_method, None, parallel, params_transition, params_observation)

    def fun_to_iter(curr_nominal_traj):
        return filter_smoother(observations, x0, transition_model, observation_model, linearization_method,
                               curr_nominal_traj, parallel, params_transition, params_observation)

    nominal_traj = fixed_point(fun_to_iter, init_nominal_trajectory, criterion)
    if return_loglikelihood:
        _, ell = filtering(observations, x0, transition_model, observation_model, linearization_method,
                           nominal_traj, parallel, return_loglikelihood=True, params_transition=params_transition, params_observation=params_observation)
        return nominal_traj, ell
    return nominal_traj


def sampling(key: jnp.ndarray,
             n_samples: int,
             transition_model: Union[FunctionalModel, ConditionalMomentsModel],
             filter_trajectory: MVNSqrt or MVNStandard,
             linearization_method: Callable,
             nominal_trajectory: Optional[Union[MVNSqrt, MVNStandard]] = None,
             parallel: bool = True,
             params_transition: Tuple = None):
    nominal_trajectory = nominal_trajectory or smoothing(transition_model, filter_trajectory, linearization_method,
                                                         None, parallel, params_transition)
    if parallel:
        return _par_sampling(key, n_samples, transition_model, filter_trajectory, linearization_method,
                             nominal_trajectory, params_transition)
    return _seq_sampling(key, n_samples, transition_model, filter_trajectory, linearization_method, nominal_trajectory, params_transition)
