from abc import ABC, abstractmethod
from typing import Iterable, Callable, NamedTuple, Tuple
import functools

import numpy as np
import jax.numpy as jnp
import jax
import numpyro.distributions as dist

from .autoregressive_utils import Context
from .training import Dataset


def create_simulator(simulator_fn, param_names):
    """Make simulator function handle thinning of samples during SNL"""

    def simulate(params, num_samples=None, **sim_kwargs):
        if num_samples is not None:
            thinned_params = thin(params, num_samples)
        else:
            thinned_params = params
        filtered_params = {param: thinned_params[param] for param in param_names}
        kwargs = {**filtered_params, **sim_kwargs}
        return simulator_fn(**kwargs)

    simulate._is_simulator = True
    return simulate


def thin(params, num_samples):
    """Thin params as much as possible to obtain a specified number of samples"""
    thinned_params = {}
    for param in params:
        param_samples = params[param]
        if param_samples.shape[0] >= num_samples:
            thinning_amount = param_samples.shape[0] // num_samples
            thinned_params[param] = param_samples[::thinning_amount][:num_samples]
        elif param_samples.shape[0] < num_samples:
            # thinned_params[param] = param_samples
            raise IndexError(
                "num_samples is greater than the total number of parameters given"
            )

    return thinned_params


class Inference(ABC):

    def __init__(self, observation, simulator, obs_name="obs", param_names=None):
        self._obs = observation
        try:
            assert simulator._is_simulator
        except:
            assert callable(simulator)
            assert (
                param_names is not None
            ), """You must either create the simulator function using create_simulator, 
            or pass the parameter names required for the simulator to the param_names argument."""
            self._pickle_simulator = simulator  # Required for pickling
            self._param_names = param_names
            simulator = create_simulator(simulator, param_names)

        self._simulator = simulator
        self._obs_name = obs_name
        self._diagnostics = []
        self._samples = []

    def to_state_dict(self):
        state_dict = {
            "observation": self._obs,
            "simulator": self._pickle_simulator,
            "obs_name": self._obs_name,
            "diagnostics": self._diagnostics,
            "samples": self._samples,
            "param_names": self._param_names,
        }
        return state_dict

    @property
    def obs(self):
        return self._obs

    @property
    def simulator(self):
        return self._simulator

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def samples(self):
        return self._samples

    def set_observation(self, obs):
        self._obs = obs

    @abstractmethod
    def step(self):
        """Apply a step of the inference algorithm."""
        pass


def train_val_split(dataset, validation_prop):
    """Training validation split. Works for datasets of the Dataset class."""

    num_samples = dataset.data.shape[0]
    indices = np.random.permutation(num_samples)
    train_indices = indices[int(validation_prop * num_samples) :]
    validation_indices = indices[: int(validation_prop * num_samples)]

    return dataset[train_indices], dataset[validation_indices]


def subsample_dataset(dataset, num_samples):
    """Randomly sample num_samples samples from a dataset"""
    size = dataset.data.shape[0]
    indices = np.random.choice(np.arange(size), size=num_samples, replace=False)
    return dataset[indices]


def merge_datasets(dataset, new_dataset, max_dataset_size):
    """Merge two datasets into a single dataset"""
    new_data = new_dataset.data
    new_context = new_dataset.context
    if (
        max_dataset_size is not None
        and new_data.shape[0] + dataset.data.shape[0] > max_dataset_size
    ):
        dataset = subsample_dataset(dataset, max_dataset_size - new_data.shape[0])
    if isinstance(new_context, Context):
        new_params = new_context.params
        new_mask = new_context.mask
        new_events = new_context.events
        new_left = new_context.left_support
        new_right = new_context.right_support

        merged_params = np.concatenate((new_params, dataset.context.params), axis=0)
        assert type(new_mask) == type(dataset.context.mask)
        merged_masks = (
            np.concatenate((new_mask, dataset.context.mask), axis=0)
            if new_mask is not None
            else None
        )
        assert type(new_events) == type(dataset.context.events)
        merged_events = (
            np.concatenate((new_events, dataset.context.events), axis=0)
            if new_events is not None
            else None
        )
        assert type(new_left) == type(dataset.context.left_support)
        merged_left = (
            np.concatenate((new_left, dataset.context.left_support))
            if isinstance(new_left, Iterable)
            else new_left
        )
        assert type(new_right) == type(dataset.context.right_support)
        merged_right = (
            np.concatenate((new_right, dataset.context.right_support))
            if isinstance(new_right, Iterable)
            else new_right
        )

        merged_context = Context(
            params=merged_params,
            events=merged_events,
            mask=merged_masks,
            left_support=merged_left,
            right_support=merged_right,
        )
    else:
        merged_context = np.concatenate(dataset.context, new_context)

    return Dataset(
        data=jnp.concatenate((new_data, dataset.data), axis=0), context=merged_context
    )


def freeze_logpdf(logpdf_fn, state):
    """Return logpdf of autoregressive model with frozen neural network parameters"""

    @jax.jit
    def frozen_logpdf(inputs, context):
        return logpdf_fn(state.params, inputs=inputs, context=context)

    return frozen_logpdf


def freeze_dist(autoregressive, state, event_shape):
    logpdf_fn = autoregressive.logpdf
    frozen_logpdf = freeze_logpdf(logpdf_fn, state)
    return NumpyroContainer(logpdf_fn=frozen_logpdf, event_shape=event_shape)


class NumpyroContainer(NamedTuple):
    logpdf_fn: Callable
    event_shape: Tuple[int]


class AutoregressiveDistribution(dist.Distribution):

    def __init__(self, base, context, init_vals=None):
        super(AutoregressiveDistribution, self).__init__(event_shape=base.event_shape)
        self.logpdf_fn = base.logpdf_fn
        self.context = context
        if init_vals is None:
            self.init_vals = jnp.ones(context.params.shape)
        else:
            self.init_vals = init_vals

    def sample(self, key, sample_shape=()):
        return jnp.ones((*sample_shape, *self.event_shape))

    def log_prob(self, x):
        return jnp.sum(self.logpdf_fn(x, self.context))


def to_numpyro(model, state, event_shape):
    container = freeze_dist(model, state, event_shape)
    return functools.partial(AutoregressiveDistribution, base=container)


def stack_and_broadcast(*args, broadcast_to=None):
    if broadcast_to is None:
        num_repeats = 1
    else:
        num_repeats = broadcast_to.shape[0]
    return jnp.repeat(
        jnp.stack(jax.tree_map(lambda x: x.reshape(1, -1), args), axis=-1).reshape(
            1, -1
        ),
        repeats=num_repeats,
        axis=0,
    )
