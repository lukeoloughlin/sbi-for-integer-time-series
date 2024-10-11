import os
import argparse
import time
import sys

sys.path.append("/home/luke/SBI")


import dill
import numpy as np
import numba
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


from src import (
    DiscreteAutoregressiveModel,
    NetworkConfig,
    ParamConfig,
    Context,
    Dataset,
    DistributionModel,
    stack_and_broadcast,
    SNL,
    SNLConfig,
    make_folder,
    Logger,
)

from experiments.simulators import SIR


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import os
import argparse
import time

import dill
import numpy as np
import numba
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import src.lfi as lfi
import src.training as train
import src.utils as util
from src import (
    DiscreteAutoregressiveModel,
    NetworkConfig,
    ParamConfig,
    Context,
)
from src.training.training import Dataset

from experiments.simulators import SIR


class TruncatedGamma(dist.Distribution):
    arg_constraints = {
        "concentration": dist.constraints.positive,
        "rate": dist.constraints.positive,
        "low": dist.constraints.positive,
    }
    support = dist.constraints.positive
    reparametrized_params = ["concentration", "rate", "low"]

    def __init__(self, concentration, rate=1.0, low=0.0, *, validate_args=None):
        self.concentration, self.rate, self.low = dist.util.promote_shapes(
            concentration, rate, low
        )
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(concentration), jnp.shape(rate), jnp.shape(low)
        )
        super(TruncatedGamma, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        pass

    @dist.util.validate_sample
    def log_prob(self, value):
        return (self.concentration - 1) * jnp.log(value) - self.rate * value


def create_likelihood_model(args):
    def event_mask_fn(x, context, *_args):
        N = context.right_support.reshape(-1, 1, 1)
        mask = x != 1
        midpoints_mask = (x[:, 1:, :] - x[:, :-1, :]) != 0
        mask = mask.at[:, 1:, :].set(midpoints_mask)
        mask = mask * (x != N)
        return mask

    network_config = NetworkConfig(
        kernel_shape=args.kernel_length,
        hidden_channels=args.hidden_channels,
        residual_blocks=args.residual_blocks,
        activation=nn.gelu if args.dropout_prob == 0.0 else nn.relu,
    )
    param_config = ParamConfig(
        encode_dim=args.param_encode_dim,
        hidden_dim=args.param_hidden_dim,
        activation=nn.gelu,
    )
    model = DiscreteAutoregressiveModel(
        network_config=network_config,
        param_config=param_config,
        monotonic=True,
        mixture_components=args.mixture_components,
        event_mask_fn=event_mask_fn,
        epidemic=True,
    )

    return train.DistributionModel(
        model,
        optimizer=optax.adamw,
        optimizer_kwargs={"learning_rate": args.lr, "weight_decay": args.weight_decay},
    )


def numpyro_SIR(likelihood, obs=None):
    R0 = numpyro.sample("R0", dist.Uniform(0, 10))
    gamma_inv = numpyro.sample("gamma_inv", dist.Gamma(10, 2))

    non_trivial_events = (
        obs.context.events[:, 0] != 0
    )  # Find observations which don't fade out immediately
    p = R0 / (R0 + 1)
    numpyro.sample("die_out_obs", dist.Bernoulli(probs=p), obs=non_trivial_events)

    obs = obs[non_trivial_events]
    params = lfi.stack_and_broadcast(
        R0 / gamma_inv, 1 / gamma_inv, broadcast_to=obs.data
    )
    context = Context(
        params=params,
        mask=obs.context.mask,
        events=obs.context.events,
        left_support=obs.context.left_support,
        right_support=obs.context.right_support,
    )
    numpyro.sample("obs", likelihood(context=context), obs=obs.data)


##########################################################################################


def create_snl(args, model, observation, numpyro_model, simulator):
    snl = lfi.SNL(
        model,
        numpyro_model,
        observation,
        simulator,
        param_names=["R0", "gamma_inv"],
        max_dataset_size=args.max_dataset_size,
    )
    snl_config = lfi.SNLConfig(
        mcmc_init_strategy=numpyro.infer.init_to_value(
            values={"R0": 2.0, "gamma_inv": 5.0}
        ),
        mcmc_num_chains=args.num_chains,
        train_early_stopping_threshold=args.es_threshold,
        train_num_epochs=500,
        train_batch_size=512,
    )
    return snl, snl_config


def run_snl(args, snl_object, snl_config, dataset, logger):
    label = args.label if args.label is not None else ""
    rng = jax.random.PRNGKey(hash("SIR_hh" + label))
    t0 = time.time()
    snl_object.SNL(
        rng,
        config=snl_config,
        dataset=dataset,
        max_num_rounds=args.num_rounds,
        logger=logger,
    )
    t1 = time.time()
    return snl_object, (t1 - t0)


def create_mask_and_events(I, Z1):
    die_out = I[:, -1, 0] == 0
    mask = np.ones_like(Z1)
    events = -np.ones((I.shape[0], 1))
    for i in range(I.shape[0]):
        if die_out[i]:
            mask_idx = np.where(Z1[i, :, 0] == Z1[i, -1, 0])[0][0]
            if mask_idx < (mask.shape[1] - 1):
                mask[i, (mask_idx + 1) :, 0] = 0.0
            events[i, 0] = mask_idx
    return mask, events.astype(int)


def main(args):
    assert args.num_obs in [100, 200, 500]
    util.make_folder("SIR_experiments")
    current_dir = os.path.dirname(__file__)
    label = "_" + args.label if args.label is not None else ""
    filename = os.path.join(
        current_dir, f"SIR_experiments/SIR_{args.num_obs}_hh" + label
    )
    np.random.seed(123456)
    with util.Logger(filename + ".log") as logger:

        logger.write(
            f""" 
Experiment details:
    kernel length: {args.kernel_length}
    hidden channels: {args.hidden_channels}
    hidden layers: {args.hidden_layers}
    parameter encoding dimension: {args.param_encode_dim}
    parameter encoder hidden dimension: {args.param_hidden_dim}
    number of mixture components: {args.mixture_components}
    learning rate: {args.lr}
    weight decay: {args.weight_decay}
    simulation budget: {args.max_dataset_size}
    number of mcmc chains: {args.num_chains}
    early stopping threshold: {args.es_threshold}
    prior samples: {args.init_samples}
                        
        """
        )

        with open(f"data/SIR_{args.num_obs}_hh_data.pkl", "rb") as f:
            obs = dill.load(f)

        def simulate(R0, gamma_inv):
            beta = R0 / gamma_inv
            gamma = 1 / gamma_inv

            N = np.random.choice(np.arange(2, 8).astype(int), beta.shape[0])

            Z1, Z2 = SIR(
                N=N,
                initial_infected=1,
                beta=beta,
                gamma=gamma,
                time_run=80,
                remove_trivial=True,
            )
            mask, events = create_mask_and_events(Z1 - Z2, Z1)
            params = np.stack((beta, gamma), axis=-1)
            return Dataset(
                data=Z1,
                context=Context(
                    params=params,
                    mask=mask,
                    events=events,
                    left_support=0,
                    right_support=N,
                ),
            )

        def truncated_gamma(shape, scale, ltrunc, size):
            out = np.zeros(size)
            for i in range(size):
                sample = 0.0
                while True:
                    sample = np.random.gamma(shape, scale)
                    if sample > ltrunc:
                        break
                out[i] = sample
            return out

        R0s = 10 * np.random.rand(args.init_samples)
        gamma_invs = truncated_gamma(10, 0.5, 1.0, args.init_samples)
        dataset = simulate(R0s, gamma_invs)
        model = create_likelihood_model(args)
        snl, snl_config = create_snl(args, model, obs, numpyro_SIR, simulate)
        snl, run_time = run_snl(args, snl, snl_config, dataset, logger)

        logger.write(f"\n\nTotal run time: {run_time:.4f} seconds.")

    snl.save_samples(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SNL on household observations of SIR model"
    )
    parser.add_argument("--kernel-length", default=5, type=int, help="kernel length")
    parser.add_argument("--num-obs", default=100, type=int, help="number of households")
    parser.add_argument(
        "--num-rounds", default=10, type=int, help="Number of snl rounds"
    )
    parser.add_argument("--label", default=None, type=str, help="experiment label")
    parser.add_argument(
        "--hidden-channels", default=150, type=int, help="num hidden channels"
    )
    parser.add_argument(
        "--residual-blocks", default=2, type=int, help="num hidden layers"
    )
    parser.add_argument(
        "--param-encode-dim", default=12, type=int, help="encoding dim of parameters"
    )
    parser.add_argument(
        "--param-hidden-dim",
        default=64,
        type=int,
        help="hidden dim of parameter encoder",
    )
    parser.add_argument(
        "--mixture-components",
        default=5,
        type=int,
        help="number of logistic mixture components",
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--weight-decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument(
        "--max-dataset-size", default=50000, type=int, help="simulation budget"
    )
    parser.add_argument("--num-chains", default=1, type=int, help="num mcmc chains")
    parser.add_argument(
        "--es-threshold",
        default=30,
        type=int,
        help="number of epochs before early stopping is used",
    )
    parser.add_argument(
        "--init-samples",
        default=10000,
        type=int,
        help="Number of prior samples in first round of SNL",
    )

    args = parser.parse_args()
    main(args)
