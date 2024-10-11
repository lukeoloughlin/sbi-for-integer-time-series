import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logit, logsumexp

from .autoregressive_utils import Distribution


def _logistic_logcdf(x, shift, scale):
    return -jnp.logaddexp(0, -(x - shift) / scale)


def _logistic_logccdf(a, shift, scale):
    return -(a - shift) / scale - jnp.logaddexp(-(a - shift) / scale, 0)


def _logistic_logcdfdifference(a, b, shift, scale):
    return (
        -(a - shift) / scale
        + jnp.log1p(-jnp.exp((a - b) / scale))
        - jnp.logaddexp(-(b - shift) / scale, 0)
        - jnp.logaddexp(-(a - shift) / scale, 0)
    )


def _dlogistic_logpdf(x, shift, scale):
    return _logistic_logcdfdifference(x - 0.5, x + 0.5, shift, scale)


def _logistic_logpdf(x, shift, scale):
    return (
        -(x - shift) / scale
        - jnp.log(scale)
        - 2 * jnp.log(1 + jnp.exp(-(x - shift) / scale))
    )


def _sample_truncated(rng, shift, scale, left, right, shape):
    u = jax.random.uniform(rng, shape)
    mul = jnp.exp(_logistic_logcdfdifference(left, right, shift, scale))
    u = mul * u + jnp.exp(_logistic_logcdf(left, shift, scale))
    # See Note 1.
    return jnp.where(
        jnp.logical_or(u >= 1, left == right),
        left,
        jnp.where(u <= 0, right, scale * logit(u) + shift),
    )


class Logistic(Distribution):

    def logpdf(self, x, shift, scale):
        return _logistic_logpdf(x, shift, scale)

    def logcdf(self, x, shift, scale):
        return _logistic_logcdf(x, shift, scale)

    def pdf(self, x, shift, scale):
        return jnp.exp(self.logpdf(x, shift, scale))

    def cdf(self, x, shift, scale):
        return jnp.exp(self.logcdf(x, shift, scale))

    def sample(self, rng, shift, scale, shape=None):
        if shape is None:
            assert (
                shift.shape == scale.shape
            ), "shift and scale must have the same shape"
            shape = shift.shape
        u = jax.random.uniform(rng, shape)
        return shift + scale * logit(u)


class DiscretizedLogistic(Distribution):

    def logpdf(self, x, shift, scale):
        return _dlogistic_logpdf(x, shift, scale)

    def pdf(self, x, shift, scale):
        return jnp.exp(self.logpdf(x, shift, scale))

    def sample(self, rng, shift, scale, shape=None):
        if shape is None:
            assert (
                shift.shape == scale.shape
            ), "shift and scale must have the same shape"
            shape = shift.shape
        u = jax.random.uniform(rng, shape)
        return jnp.round(shift + scale * logit(u))


class TruncatedLogistic(Distribution):

    def logpdf(self, x, shift, scale, left, right):
        return _logistic_logpdf(x, shift, scale) - _logistic_logcdfdifference(
            left, right, shift, scale
        )

    def pdf(self, x, shift, scale, left, right):
        return jnp.exp(self.logpdf(x, shift, scale, left, right))

    def sample(self, rng, shift, scale, left, right, shape=None):
        if shape is None:
            assert (
                shift.shape == scale.shape
            ), "shift and scale must have the same shape"
            shape = shift.shape
        return _sample_truncated(rng, shift, scale, left, right, shape)


class TruncatedDiscretizedLogistic(Distribution):
    unbounded_right: bool = False

    def logpdf(self, x, shift, scale, left, right):
        log_num = _logistic_logcdfdifference(x, x + 1, shift, scale)
        if not self.unbounded_right:
            log_denom = _logistic_logcdfdifference(left, right + 1, shift, scale)
        else:  # See NOTE 2.
            log_denom = _logistic_logccdf(left, shift, scale)
        return log_num - log_denom

    def pdf(self, x, shift, scale, left, right):
        return jnp.exp(self.logpdf(x, shift, scale, left, right))

    def sample(self, rng, shift, scale, left, right, shape=None):
        if shape is None:
            assert (
                shift.shape == scale.shape
            ), "shift and scale must have the same shape"
            shape = shift.shape
        x = _sample_truncated(rng, shift, scale, left, right + 1, shape)
        return jnp.floor(x)


class MixtureLogistic(Distribution):
    base: Distribution = Logistic()

    def setup(self):
        self.is_truncated = isinstance(
            self.base, (TruncatedLogistic, TruncatedDiscretizedLogistic)
        )

    def logpdf(self, x, shifts, scales, log_proportions, left=None, right=None):
        assert (
            shifts.shape == scales.shape and shifts.shape == log_proportions.shape
        ), "Must input the same number of shift, scale and proportion parameters"
        out = []
        for i in range(shifts.shape[-1]):
            if self.is_truncated:
                out.append(
                    log_proportions[..., i]
                    + self.base.logpdf(x, shifts[..., i], scales[..., i], left, right)
                )
            else:
                out.append(
                    log_proportions[..., i]
                    + self.base.logpdf(x, shifts[..., i], scales[..., i])
                )
        out = jnp.stack(out, axis=-1)
        return logsumexp(out, axis=-1)

    def pdf(self, x, shifts, scales, proportions):
        return jnp.exp(self.logpdf(x, shifts, scales, proportions))

    def sample(
        self,
        rng,
        shifts,
        scales,
        proportions,
        left=None,
        right=None,
        shape=None,
        use_logits=True,
    ):
        assert (
            shifts.shape == scales.shape and shifts.shape == proportions.shape
        ), "Must input the same number of shift, scale and proportion parameters"
        if not use_logits:
            proportions = logit(proportions)
        category_rng, rng = jax.random.split(rng)
        if shape is None:
            batch_size = shifts.shape[0]
            categories = jax.random.categorical(category_rng, proportions)
            shifts = shifts[np.arange(batch_size), categories]
            scales = scales[np.arange(batch_size), categories]
        else:
            categories = jax.random.categorical(category_rng, proportions, shape=shape)
            shifts = shifts[categories]
            scales = scales[categories]
        if self.is_truncated:
            return self.base.sample(rng, shifts, scales, left, right, shape)
        else:
            return self.base.sample(rng, shifts, scales, shape)


########################################################################################################################
# NOTES
########################################################################################################################

# NOTE 1: If u (after shift and scaling) comes out to be 1, then all of the probability mass is put onto the left point.
# This happens when shift < left and scale is small

# NOTE 2: Using the log ccdf of the logistic is equivalent to using the logistic log cdf difference with b=jnp.inf,
# however even though these two functions return the same result, using jnp.inf creates a problem when calculating
# gradients, so we have to work around it
