import os
import argparse
import time
import re

import dill
import numpy as np
import numba
import numpyro
import numpyro.distributions as dist
import jax
import flax.linen as nn
import optax
import jax.numpy as jnp

import src.lfi as lfi
import src.training as train
import src.utils as util
from src.distributions import (
    DiscreteConvMADE,
    NetworkConfig,
    ParamConfig,
    SupportConfig,
    Context,
)
from src.training.training import Dataset

from experiments.simulators import LV_immigration


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def create_likelihood_model(args):

    network_config = NetworkConfig(
        kernel_shape=args.kernel_length,
        hidden_channels=args.hidden_channels,
        hidden_layers=args.hidden_layers,
        data_scale=900,
        activation=nn.gelu if args.dropout_prob == 0.0 else nn.relu,
    )

    param_config = ParamConfig(
        encode_dim=args.param_encode_dim,
        hidden_dim=args.param_hidden_dim,
        activation=nn.gelu,
    )
    support_config = SupportConfig(unbounded_upper_support=True)

    model = DiscreteConvMADE(
        network_config=network_config,
        param_config=param_config,
        support_config=support_config,
        mixture_components=args.mixture_components,
    )

    return train.DistributionModel(
        model,
        optimizer=optax.adamw,
        optimizer_kwargs={"learning_rate": args.lr, "weight_decay": args.weight_decay},
    )


def numpyro_LV_immigration(likelihood, obs=None):
    log_alpha = numpyro.sample("log_alpha", dist.Normal(jnp.log(0.1), 0.5))
    log_beta = numpyro.sample("log_beta", dist.Normal(jnp.log(0.002), 0.5))
    log_gamma = numpyro.sample("log_gamma", dist.Normal(jnp.log(0.2), 0.5))
    log_delta = numpyro.sample("log_delta", dist.Normal(jnp.log(0.0025), 0.5))

    lambda_1 = numpyro.sample("lambda_1", dist.Uniform(0.1, 25))
    lambda_2 = numpyro.sample("lambda_2", dist.Uniform(0.1, 25))

    alpha = numpyro.deterministic("alpha", jnp.exp(log_alpha))
    beta = numpyro.deterministic("beta", jnp.exp(log_beta))
    gamma = numpyro.deterministic("gamma", jnp.exp(log_gamma))
    delta = numpyro.deterministic("delta", jnp.exp(log_delta))

    params = lfi.stack_and_broadcast(
        log_alpha,
        log_beta,
        log_gamma,
        log_delta,
        jnp.log(lambda_1),
        jnp.log(lambda_2),
        broadcast_to=obs.data,
    )

    context = Context(
        params=params,
        mask=obs.context.mask,
        events=obs.context.events,
        left_support=obs.context.left_support,
        right_support=obs.context.right_support,
    )
    numpyro.sample("obs", likelihood(context=context), obs=obs.data)


def create_snl(args, model, observation, numpyro_model, prior_sampler, simulator):
    snl = lfi.SNL(
        model,
        numpyro_model,
        observation,
        simulator,
        prior_sampler,
        param_names=[
            "log_alpha",
            "log_beta",
            "log_gamma",
            "log_delta",
            "lambda_1",
            "lambda_2",
        ],
        max_dataset_size=args.max_dataset_size,
    )
    snl_config = lfi.SNLConfig(
        mcmc_init_strategy=numpyro.infer.init_to_median(num_samples=15),
        mcmc_num_chains=args.num_chains,
        train_early_stopping_threshold=args.es_threshold,
        train_num_epochs=700,
        train_batch_size=512,
    )
    return snl, snl_config


def run_snl(snl_object, snl_config, dataset, logger):
    label = args.label if args.label is not None else ""
    rng = jax.random.PRNGKey(hash("LV_immigration" + label))
    snl_object.set_diagnostic_parameter(
        log_alpha=np.log(0.1),
        log_beta=np.log(0.002),
        log_gamma=np.log(0.2),
        log_delta=np.log(0.0025),
        lambda_1=10,
        lambda_2=10,
    )
    t0 = time.time()
    snl_object.SNL(
        rng,
        config=snl_config,
        dataset=dataset,
        max_num_rounds=args.num_rounds,
        terminate_early=False,
        logger=logger,
    )
    t1 = time.time()
    return snl_object, (t1 - t0)


def main(args):
    util.make_folder("LV_immigration_experiments")
    current_dir = os.path.dirname(__file__)
    label = "_" + args.label if args.label is not None else ""
    filename = os.path.join(current_dir, f"LV_immigration_experiments/LV_imm" + label)
    np.random.seed(12345)
    with util.Logger(filename + ".log") as logger:

        logger.write(
            f""" 
Experiment details:
    kernel length: {args.kernel_length}
    hidden channels: {args.hidden_channels}
    hidden layers: {args.hidden_layers}
    activation: {args.activation}
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
        with open(f"data/LV_imm_1_obs_data.pkl", "rb") as f:
            obs = dill.load(f)

        def simulate(log_alpha, log_beta, log_gamma, log_delta, lambda_1, lambda_2):
            n_trials = log_alpha.shape[0]
            init_pred_noise = obs.data[0, 0, 0].astype(int)
            init_prey_noise = obs.data[0, 0, 1].astype(int)

            # Numpy counts the number of failures, not successes, so we use p not 1 - p
            init_prey = (
                np.random.negative_binomial(init_prey_noise + 1, p=0.9, size=n_trials)
                + init_prey_noise
            )
            init_pred = (
                np.random.negative_binomial(init_pred_noise + 1, p=0.9, size=n_trials)
                + init_pred_noise
            )

            alpha, beta, gamma, delta = (
                np.exp(log_alpha),
                np.exp(log_beta),
                np.exp(log_gamma),
                np.exp(log_delta),
            )
            prey, pred = LV_immigration(
                alpha,
                beta,
                gamma,
                delta,
                lambda_1,
                lambda_2,
                initial_predator=init_pred,
                initial_prey=init_prey,
                predator_cc=1000,
                prey_cc=1000,
                time_run=100,
            )

            pred[:, 0, 0] = init_pred_noise
            prey[:, 0, 0] = init_prey_noise
            pred[:, 1:, 0] = np.random.binomial(pred[:, 1:, 0].astype(int), 0.9).astype(
                float
            )
            prey[:, 1:, 0] = np.random.binomial(prey[:, 1:, 0].astype(int), 0.9).astype(
                float
            )
            params = np.stack(
                (
                    log_alpha,
                    log_beta,
                    log_gamma,
                    log_delta,
                    np.log(lambda_1),
                    np.log(lambda_2),
                ),
                axis=-1,
            )
            data = np.concatenate((pred, prey), axis=-1)
            return Dataset(
                data=data,
                context=Context(
                    params=params,
                    mask=None,
                    events=None,
                    left_support=0,
                    right_support=None,
                ),
            )

        def prior(num_samples):
            return {
                "log_alpha": np.log(0.1) + 0.5 * np.random.normal(size=num_samples),
                "log_beta": np.log(0.002) + 0.5 * np.random.normal(size=num_samples),
                "log_gamma": np.log(0.2) + 0.5 * np.random.normal(size=num_samples),
                "log_delta": np.log(0.0025) + 0.5 * np.random.normal(size=num_samples),
                "lambda_1": 0.1 + 24.9 * np.random.rand(size=num_samples),
                "lambda_2": 0.1 + 24.9 * np.random.rand(size=num_samples),
            }

        log_alphas = np.log(0.1) + 0.5 * np.random.normal(size=args.init_samples)
        log_betas = np.log(0.002) + 0.5 * np.random.normal(size=args.init_samples)
        log_gammas = np.log(0.2) + 0.5 * np.random.normal(size=args.init_samples)
        log_deltas = np.log(0.0025) + 0.5 * np.random.normal(size=args.init_samples)
        lambda_1s = 0.1 + (25 - 0.1) * np.random.rand(args.init_samples)
        lambda_2s = 0.1 + (25 - 0.1) * np.random.rand(args.init_samples)
        dataset = simulate(
            log_alphas, log_betas, log_gammas, log_deltas, lambda_1s, lambda_2s
        )
        model = create_likelihood_model(args)
        snl, snl_config = create_snl(
            args, model, obs, numpyro_LV_immigration, prior, simulate
        )
        snl, run_time = run_snl(snl, snl_config, dataset, logger)

        logger.write(f"\n\nTotal run time: {run_time:.4f} seconds.")

    snl.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNL on LV model")
    parser.add_argument("--kernel-length", default=5, type=int, help="kernel length")
    parser.add_argument("--label", default=None, type=str, help="experiment label")
    parser.add_argument(
        "--num-rounds", default=10, type=int, help="Number of snl rounds"
    )
    parser.add_argument(
        "--dropout-prob", default=0.0, type=float, help="experiment label"
    )
    parser.add_argument(
        "--hidden-channels", default=256, type=int, help="num hidden channels"
    )
    parser.add_argument(
        "--hidden-layers", default=5, type=int, help="num hidden layers"
    )
    parser.add_argument("--activation", default="gelu", type=str, help="activation fn")
    parser.add_argument(
        "--param-encode-dim", default=24, type=int, help="encoding dim of parameters"
    )
    parser.add_argument(
        "--param-hidden-dim",
        default=128,
        type=int,
        help="hidden dim of parameter encoder",
    )
    parser.add_argument(
        "--mixture-components",
        default=5,
        type=int,
        help="number of logistic mixture components",
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
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
