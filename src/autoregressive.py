import functools
from typing import Any, Callable, Optional, Tuple, Iterable, Sequence, Union

import numpy as np
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import normal, zeros

from .logistic import DiscretizedLogistic, TruncatedDiscretizedLogistic, MixtureLogistic
from .autoregressive_utils import (
    ParamConfig,
    NetworkConfig,
    ContextEncoder,
    calculate_dmol_params,
    calculate_logpdf,
    unpack_context,
)

from .autoregressive_utils import causal_conv_masks

PRNGKey = Shape = Dtype = Array = Any
default_kernel_init = normal(1e-6)
Shape = Iterable[int]


########################################################################################################################
# Causal CNN
########################################################################################################################


def conv_dimension_numbers(input_shape):
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class MaskedConv(nn.Module):
    mask: jnp.ndarray
    strides: Union[None, int, Iterable[int]] = 1
    padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
    input_dilation: Union[None, int, Iterable[int]] = 1
    kernel_dilation: Union[None, int, Iterable[int]] = 1
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        assert isinstance(
            self.mask, (np.ndarray, jnp.ndarray)
        ), "Mask should be a jnp.ndarray"
        inputs = jnp.asarray(inputs, self.dtype)

        kernel_size = self.mask.shape[:-2]

        def maybe_broadcast(x):
            if x is None:
                x = 1

            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return x

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        kernel = self.param("kernel", self.kernel_init, self.mask.shape)
        kernel = jnp.asarray(kernel, self.dtype)
        dn = conv_dimension_numbers(inputs.shape)
        y = lax.conv_general_dilated(
            inputs,
            kernel * self.mask,
            strides,
            self.padding,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dn,
            precision=self.precision,
        )
        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.mask.shape[-1],))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


def _CausalCNN_setup(
    mdl: nn.Module,
    inputs: Array,
):
    """Create the layers and masks for the CNN."""
    out_param_dims = [mdl.out_param_dimension] * inputs.shape[-1]

    kernel_shape = (mdl.kernel_shape, inputs.shape[-1], mdl.hidden_channels)
    padding = [(mdl.kernel_shape - 1, 0)]  # Left padding to make convs causal
    conv_masks = causal_conv_masks(
        kernel_shape, mdl.residual_blocks * 2, out_param_dims, epidemic=mdl.epidemic
    )
    layers = [
        functools.partial(MaskedConv, padding=padding, mask=mask) for mask in conv_masks
    ]
    return layers


def _CausalCNN_forward_pass(
    mdl: nn.Module,
    inputs: Array,
    encoded_params: Array | None,
    layers: Sequence[nn.Module],
) -> Array:
    """Evaluates the CNN"""
    outputs = layers[0](
        kernel_init=mdl.kernel_init,
        bias_init=mdl.bias_init,
        precision=mdl.precision,
        dtype=mdl.dtype,
    )(inputs)

    # Input layer
    if encoded_params is not None:
        c = nn.Dense(features=outputs.shape[-1], use_bias=False)(encoded_params)
        outputs = outputs + c.reshape(-1, 1, c.shape[-1])

    # Residual blocks
    for i, layer in enumerate(layers[1:-1]):
        if i % 2 == 0:
            residual = outputs
        outputs = mdl.activation(outputs)
        outputs = layer(
            kernel_init=mdl.kernel_init,
            bias_init=mdl.bias_init,
            precision=mdl.precision,
            dtype=mdl.dtype,
        )(outputs)

        if encoded_params is not None:
            c = nn.Dense(features=outputs.shape[-1], use_bias=False)(encoded_params)
            outputs = outputs + c.reshape(-1, 1, c.shape[-1])

        if i % 2 == 1:
            outputs = outputs + residual

    # Output layer
    outputs = mdl.activation(outputs)
    outputs = layers[-1](
        kernel_init=zeros if mdl.init_final_conv_to_zeros else mdl.kernel_init,
        precision=mdl.precision,
        dtype=mdl.dtype,
    )(outputs)

    if encoded_params is not None:
        c = nn.Dense(features=outputs.shape[-1], use_bias=False)(encoded_params)
        outputs = outputs + c.reshape(-1, 1, c.shape[-1])

    return outputs


class CausalCNN(nn.Module):
    hidden_channels: int
    residual_blocks: int
    kernel_shape: int = 2
    out_param_dimension: int = 2
    activation: Callable = nn.relu
    epidemic: bool = False
    init_final_conv_to_zeros: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    """
    CNN with causal convolutions and MADE-like masks along the channel dimension.

    hidden_channels: Number of hidden channels in each hidden layer.
    residual_blocks: Number of residual blocks.
    kernel_shape: length of convolution kernel.
    out_param_dimension: Number of output channels.
    num_events: Number of inputs which are binary.
    activation: Activation function for hidden layers.
    epidemic: True if running outbreak experiments.
    events_pos: If binary events are input to the network, then this specifies how they are ordered in the input.
    init_final_conv_to_zeros: If True, the kernel of the final layer is initialised to zeros.
    """

    @nn.compact
    def __call__(self, inputs, context=None):
        layers = _CausalCNN_setup(self, inputs)
        outputs = _CausalCNN_forward_pass(self, inputs, context, layers)
        return outputs


########################################################################################################################
# Autoregressive Model
########################################################################################################################


class DiscreteAutoregressiveModel(nn.Module):
    network_config: NetworkConfig
    mixture_components: int
    monotonic: bool = False
    param_encoder: nn.Module = ContextEncoder
    param_config: Optional[ParamConfig] = ParamConfig()
    unbounded_support: bool = False
    event_mask_fn: Optional[Callable] = None
    epidemic: bool = False
    rescale_dmol_params: bool = True

    """Autoregressive model for discrete time series data. 
    
    network_config: Essentially a named tuple of CNN hyperparams.
    mixture_components: Number of DMoL mixture components.
    monotonic: True if data is increasing. Truncates the lower support of q(y_i|y_{1:i-1}) to y_{i-1}.
    param_encoder: Neural net to calculate c(theta)
    param_config: Essentially a named tuple of hyperparams for context shallow neural net. Only needs to be passed if default for param_encoder is
    used
    unbounded_support: True if upper support is inf. Prevents an annoying error from occuring from JAX tracers. 
    event_mask_fn: A function to calculate appropriate masks when binary indicator log probs can be zeroed. Should be of the form f(inputs, ctx), so
    that the masking can be effected by y_{1:n}, and the context object holding (theta, mask, support). 
    epidemic: True if running outbreak experiments. Handles the additional output channel in the CNN and implements appropriate reshifting/rescaling
    of logistic shift and scale params.
    events_pos: Position that indicators should come in input sequence. 
    rescale_dmol_params: If True and y_{1:n} -> y_{1:n} / C before inputting to the CNN, then the shifts (before adding y_{i-1}) and scales are
    multiplied by C.
    """

    @nn.compact
    def __call__(self, inputs, _method="logpdf", context=None):  # , _num_events=None):
        inputs = jnp.array(inputs)
        assert _method == "logpdf" or _method == "params"

        if context is not None:
            model_params, events, mask, support = unpack_context(context, inputs)
        else:
            model_params = events = mask = support = None

        conditioner = CausalCNN(
            hidden_channels=self.network_config.hidden_channels,
            residual_blocks=self.network_config.residual_blocks,
            kernel_shape=self.network_config.kernel_shape,
            out_param_dimension=(
                3 * self.mixture_components
                if not self.epidemic
                else 3 * self.mixture_components + 1
            ),
            activation=self.network_config.activation,
            epidemic=self.epidemic,
            init_final_conv_to_zeros=self.network_config.init_final_conv_to_zeros,
            dtype=self.network_config.dtype,
            precision=self.network_config.precision,
            kernel_init=self.network_config.kernel_init,
            bias_init=self.network_config.bias_init,
        )

        _, _, features = inputs.shape

        if support is None:
            base = DiscretizedLogistic()
        else:
            base = TruncatedDiscretizedLogistic(unbounded_right=self.unbounded_support)

        if self.mixture_components == 1:
            dist = base
        else:
            dist = MixtureLogistic(base=base)

        # Calculate DMoL params and binary event log probabilities
        encoded_params = self.param_encoder(self.param_config)(model_params)
        shifts, scales, proportions, log_p, log_one_take_p = calculate_dmol_params(
            self,
            conditioner,
            inputs,
            encoded_params,
            features,
            num_events=1 if self.epidemic else 0,
        )

        # Postprocess shift and scale params. If epidemic=True, exploit shrinking support. Otherwise, just rescale by the inverse of the input scaling
        # factor.
        if self.rescale_dmol_params and not self.epidemic:
            if self.network_config.data_scale is not None:
                shifts = shifts * self.network_config.data_scale
                scales = scales * self.network_config.data_scale
                shifts = shifts.at[:, 1:, :].add(inputs[:, :-1, :, jnp.newaxis])
        elif self.epidemic:
            # Shift from last obs can shrink with support
            shifts = shifts.at[:, 1:, :].multiply(
                support[1][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
                - inputs[:, :-1, :, jnp.newaxis]
            )
            # Move up to last obs
            shifts = shifts.at[:, 1:, :].add(inputs[:, :-1, :, jnp.newaxis])
            # Scales can shrink as support shrinks
            scales = scales.at[:, 1:, :].multiply(
                support[1][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
                - inputs[:, :-1, :, jnp.newaxis]
            )

        scales = scales + 1e-6  # numerical stability

        if _method == "logpdf":
            log_prob = calculate_logpdf(
                inputs,
                dist,
                shifts,
                scales,
                proportions,
                mask,
                support,
                monotonic=self.monotonic,
            )

            if self.epidemic > 0:
                event_log_probs = events * log_p + (1 - events) * log_one_take_p
                # Zero out necessary log probs for indicators
                if self.event_mask_fn is not None:
                    event_mask = self.event_mask_fn(inputs, context, mask)
                    event_log_probs = event_log_probs * event_mask
                sum_event_log_probs = jnp.sum(event_log_probs, axis=(1, 2))
            else:
                sum_event_log_probs = 0

            return jnp.sum(log_prob, axis=(1, 2)) + sum_event_log_probs

        elif _method == "params":
            if self.epidemic > 0:
                event_log_probs = events * log_p + (1 - events) * log_one_take_p
                if self.event_mask_fn is not None:
                    event_mask = self.event_mask_fn(inputs, context, mask)
                    event_log_probs = event_log_probs * event_mask
                return shifts, scales, proportions, event_log_probs
            else:
                return shifts, scales, proportions
