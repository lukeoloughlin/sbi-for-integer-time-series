from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, NamedTuple, Tuple, Sequence, Iterable
import functools

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

PRNGKey = Shape = Dtype = Array = Any

default_kernel_init = nn.initializers.normal(1e-6)

Config = NamedTuple


class Context(NamedTuple):
    """Holds context/conditioning arguments for autoregressive model"""

    params: Array
    mask: Optional[Array] = None
    events: Optional[Array] = None
    left_support: Optional[Array] = None
    right_support: Optional[Array] = None

    def __getitem__(self, idx):
        params_ = self.params[idx] if self.params is not None else None
        mask_ = self.mask[idx] if self.mask is not None else None
        events_ = self.events[idx] if self.events is not None else None
        if self.left_support is not None and not isinstance(
            self.left_support, (int, float)
        ):
            left_support_ = (
                self.left_support[idx]
                if self.left_support.shape != ()
                else self.left_support
            )
        else:
            left_support_ = self.left_support

        if self.right_support is not None and not isinstance(
            self.right_support, (int, float)
        ):
            right_support_ = (
                self.right_support[idx]
                if self.right_support.shape != ()
                else self.right_support
            )
        else:
            right_support_ = self.right_support

        return Context(
            params=params_,
            mask=mask_,
            events=events_,
            left_support=left_support_,
            right_support=right_support_,
        )


class NetworkConfig(Config):
    hidden_channels: int
    residual_blocks: int
    kernel_shape: Optional[int] = None
    data_shift: Optional[float] = None
    data_scale: Optional[float] = None
    activation: Callable = nn.gelu
    init_final_conv_to_zeros: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros


class ParamConfig(Config):
    """Configuration for parameter specific hyperparameters.
    params:
        output_dim: Size of embedding calculated from parameters. If None, then it is up to the encoding network to
        choose a default size.
        hidden_dim: Size of hidden dimension in network for computing the embedding. If None, then it is up to the encoding network to
        choose a default size.
        activation: Activation function used in network for computing embedding
    """

    encode_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    activation: Callable = nn.relu


class ContextEncoder(nn.Module):
    """Encodes real valued parameters to use as a conditioning argument in conditional distribution models."""

    config: ParamConfig

    @nn.compact
    def __call__(self, inputs):
        outputs = nn.Dense(self.config.hidden_dim)(inputs)
        outputs = self.config.activation(outputs)
        outputs = nn.Dense(self.config.encode_dim)(outputs)
        return outputs


class Distribution(nn.Module, ABC):

    def __call__(self, inputs, *args, **kwargs):
        return self.logpdf(inputs, *args, **kwargs)

    @abstractmethod
    def logpdf(self, inputs, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, rng, shape, *args, **kwargs):
        pass

    def sample_logpdf(self, rng, shape, *args, **kwargs):
        out = self.sample(rng, shape, *args, **kwargs)
        lgpdf = self.logpdf(out, *args, **kwargs)
        return out, lgpdf

    def pdf(self, inputs, *args, **kwargs):
        return jnp.exp(self.logpdf(inputs, *args, **kwargs))


def causal_conv_masks(
    kernel_shape: Sequence[int],
    num_hidden_layers: int,
    num_params: Iterable[int],
    num_augmented: int = 0,
    epidemic=False,
) -> Sequence[np.ndarray]:
    """Produces the masks to uphold the autoregressive property for a MADE
    network using convolutions

    params:
        kernel_shape: The shape of the convolution kernel
        num_hidden_layers: Number of hidden layers in the causal CNN
        num_params: Number of output parameters per feature, e.g. for epidemic model experiments where infection counts and final size indicator is
        the data, then using a DMoL with m mixture components we have num_params=(3*m+1,).
        num_augmented: Number of additional features in the hidden layers corresponding to parameters, and are consequently not masked.
        epidemic: True for the epidemic models with final size data, false otherwise. Alters the masks to reflect conditional independence properties
        of the final size indicators
    output:
        A list of the mask matrices that when pointwise multiplied with the convolution kernel in each layer of the CNN ensures that the
        autoregressive property is satisfied.
    """
    kernel_len, num_in_feat, num_hid_feat = kernel_shape
    assert (
        kernel_shape[-1] >= num_in_feat
    ), "The number of hidden channels must be >= the number of input channels"

    # partition hidden channels into blocks corresponding to input features
    hidden_channels_per_feature = num_hid_feat // num_in_feat
    rem = num_hid_feat % num_in_feat

    num_output = np.array(num_params).sum()

    all_masks = []

    # construct first mask
    in_mask = np.ones(kernel_shape)
    in_mask[-1, ...] = 0.0  # set last mask matrix in kernel to zeros
    one_channels_idx = 0  # First index of (transposed) block columns to set to ones
    for i in range(num_in_feat):
        in_mask[-1, i, one_channels_idx:] = 1.0
        extra_channel = 1 if i < rem else 0
        one_channels_idx += hidden_channels_per_feature + extra_channel
    all_masks.append(in_mask)

    # construct hidden layer mask
    hidden_mask = np.ones((kernel_len, num_hid_feat, num_hid_feat))
    hidden_mask[-1, ...] = 0.0
    one_channels_idx = 0
    for i in range(num_in_feat):
        extra_channel = 1 if i < rem else 0
        tmp = one_channels_idx + hidden_channels_per_feature + extra_channel
        hidden_mask[-1, one_channels_idx:tmp, one_channels_idx:] = 1.0
        one_channels_idx = tmp
    all_masks = all_masks + ([hidden_mask] * num_hidden_layers)

    # construct output layer mask
    out_mask = np.ones((kernel_len, num_hid_feat, num_output))
    out_mask[-1, ...] = 0.0

    extra_channel = 1 if rem > 0 else 0
    one_channels_idx = hidden_channels_per_feature + extra_channel
    one_channels_idx = 0
    one_out_idx = num_params[0]
    for i in range(1, num_in_feat):
        extra_channel = 1 if i < rem else 0
        tmp = one_channels_idx + hidden_channels_per_feature + extra_channel

        out_mask[-1, one_channels_idx:tmp, one_out_idx:] = 1.0

        one_channels_idx = tmp
        one_out_idx += num_params[i]

    # If epidemic, let the current value of counts influence the last output parameter (probability of termination)
    if epidemic:
        out_mask[-1, :, -1] = 1.0

    all_masks.append(out_mask)

    if num_augmented > 0:
        for i, mask in enumerate(all_masks):
            all_masks[i] = augment_mask(num_augmented, mask)

    return all_masks


def augment_mask(num_channels: int, mask: np.array):
    """Add additional input channels to masks to deal with parameters or other covariates in CNN"""

    # Concatenate a block of all zeros except in the final mask matrix of kernel, which is all ones
    augment_block = np.zeros((mask.shape[0], num_channels, mask.shape[-1]))
    augment_block[-1] = 1.0
    augmented_mask = np.concatenate((mask, augment_block), axis=1)
    return augmented_mask


def calculate_dmol_params(
    mdl: nn.Module,
    CNN: nn.Module,
    x: Array,
    context: Array,
    features: int,
    num_events: int,
) -> Tuple[Array, Array, Array]:
    """evaluates the CNN to calculate parameters of conditionals"""
    x = maybe_normalize(
        x,
        mdl.network_config.data_shift,
        mdl.network_config.data_scale,  # non_events_pos
    )

    mixture_params = CNN(x, context)

    # For dealing with binary events
    if num_events > 0:
        logit_p_ = mixture_params[..., -num_events:]
        log_p = nn.log_sigmoid(logit_p_)
        log_one_take_p = log_p - logit_p_
        mixture_params = mixture_params[..., :-num_events]
    else:
        log_p = jnp.zeros_like(x)
        log_one_take_p = jnp.zeros_like(x)

    shifts, scales, proportions = _reshape_and_split(mixture_params, features)

    # Do naive post processing to get the scale and mixture prop params, apply further post processing down the line
    scales = nn.softplus(scales)
    proportions = nn.log_softmax(proportions, axis=-1)
    return shifts, scales, proportions, log_p, log_one_take_p


def calculate_logpdf(
    x: Array,
    dist: Distribution,
    shifts: Array,
    scales: Array,
    proportions: Array,
    mask: Array,
    support: Array,
    monotonic: bool = False,
) -> Array:
    """Calculates the logpdf of discretised logistic conditionals across time series. Checks whether to truncate."""
    if support is None:
        if monotonic:
            log_prob = dist.logpdf(
                x[:, 1:, :],
                shifts[:, 1:, :],
                scales[:, 1:, :],
                proportions[:, 1:, :],
                left=x[:, :-1, :],
                right=jnp.inf,
            )
        else:
            log_prob = dist.logpdf(
                x[:, 1:, :], shifts[:, 1:, :], scales[:, 1:, :], proportions[:, 1:, :]
            )
    else:
        n_samples = x.shape[0]
        if monotonic:
            log_prob = dist.logpdf(
                x[:, 1:, :],
                shifts[:, 1:, :],
                scales[:, 1:, :],
                proportions[:, 1:, :],
                left=x[:, :-1, :],
                right=support[1].reshape(n_samples, 1, 1),
            )
        else:
            log_prob = dist.logpdf(
                x[:, 1:, :],
                shifts[:, 1:, :],
                scales[:, 1:, :],
                proportions[:, 1:, :],
                left=support[0].reshape(n_samples, 1, 1),
                right=support[1].reshape(n_samples, 1, 1),
            )
    if mask is not None:
        log_prob = log_prob * mask[:, 1:, :]
    return log_prob


def unpack_context(context: Context, inputs: jnp.ndarray):  # , n_samples: int):
    """Pull params, events, mask and support from Context object"""
    assert isinstance(context, Context), "Context must be either a Context object"

    n_samples = inputs.shape[0]

    params = context.params
    mask = context.mask

    no_support = context.left_support is None and context.right_support is None
    if no_support:
        # In this case, no truncation of the DMoL is used
        support = None
    else:
        if context.left_support is not None:
            left_support = context.left_support
            # The () shape condition handles an annoying issue with JAX tracers during compilation
            if isinstance(left_support, (int, float)) or left_support.shape == ():
                left_support = left_support * jnp.ones((n_samples,))
        else:
            left_support = -jnp.inf

        if context.right_support is not None:
            right_support = context.right_support
            # The () shape condition handles an annoying issue with JAX tracers during compilation
            if isinstance(right_support, (int, float)) or right_support.shape == ():
                right_support = right_support * jnp.ones((n_samples,))
        else:
            right_support = jnp.inf

        support = (left_support, right_support)

    if context.events is not None:
        events = event_pos_to_binary(context.events, inputs.shape[1])
    else:
        events = None

    return params, events, mask, support


def maybe_normalize(
    x: Array,
    shift: float,
    scale: float,
) -> Array:
    """Normalize input: x -> (x - shift) / scale."""
    if shift is not None:
        x = x - shift
    if scale is not None:
        x = x / scale
    return x


def _reshape_and_split(params: Array, features: int) -> Tuple[Array, Array, Array]:
    """Takes output of parameter network and splits the shifts, scales and proportions up for each feature of the input
    data, then organises them along a new axis"""
    params = jnp.stack(jnp.split(params, features, axis=-1), axis=-2)
    return jnp.split(params, 3, axis=-1)


@functools.partial(jax.vmap, in_axes=(0, None))
def event_pos_to_binary(events, length):
    """Turn positions of events into a binary array. -1 is associated with an array of zeros"""
    binary_event = jnp.zeros((length, events.shape[-1]))
    for i in range(events.shape[-1]):
        binary_event = jax.lax.cond(
            events[i] != -1,
            lambda x: x.at[events[i], i].set(1.0),
            lambda x: x,
            binary_event,
        )
    return binary_event
