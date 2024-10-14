import os
import argparse
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


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

from experiments.simulators import PP


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def create_likelihood_model(args):

    network_config = NetworkConfig(
        kernel_shape=args.kernel_length,
        hidden_channels=args.hidden_channels,
        residual_blocks=args.residual_blocks,
        data_scale=60.0,
        activation=nn.gelu,
    )
    param_config = ParamConfig(
        encode_dim=args.param_encode_dim,
        hidden_dim=args.param_hidden_dim,
        activation=nn.gelu,
    )
    model = DiscreteAutoregressiveModel(
        network_config=network_config,
        param_config=param_config,
        monotonic=False,
        mixture_components=args.mixture_components,
        epidemic=False,
    )

    return DistributionModel(
        model,
        optimizer=optax.adamw,
        optimizer_kwargs={"learning_rate": args.lr, "weight_decay": args.weight_decay},
    )


def numpyro_PP(likelihood, obs=None):
    log_b = numpyro.sample("log_b", dist.TruncatedNormal(jnp.log(0.25), 0.25, high=0.0))
    log_d1 = numpyro.sample("log_d1", dist.TruncatedNormal(jnp.log(0.1), 0.5, high=0.0))
    log_d2 = numpyro.sample(
        "log_d2", dist.TruncatedNormal(jnp.log(0.01), 0.5, high=0.0)
    )
    p1_ = numpyro.sample("p1_", dist.Exponential(rate=1 / 0.1))
    p1 = numpyro.deterministic("p1", p1_ + 0.01)
    p2 = numpyro.sample("p2", dist.Exponential(rate=1 / 0.05))

    params = stack_and_broadcast(
        log_b * jnp.ones(1),
        log_d1 * jnp.ones(1),
        log_d2 * jnp.ones(1),
        p1 * jnp.ones(1),
        p2 * jnp.ones(1),
        broadcast_to=obs.data,
    )
    context = Context(
        params=params,
        mask=obs.context.mask,
        events=obs.context.events,
        left_support=0,
        right_support=800,
    )
    numpyro.sample("obs", likelihood(context=context), obs=obs.data)


##########################################################################################


def create_snl(args, model, observation, numpyro_model, simulator):
    snl = SNL(
        model,
        numpyro_model,
        observation,
        simulator,
        param_names=["log_b", "log_d1", "log_d2", "p1", "p2"],
        max_dataset_size=args.max_dataset_size,
    )
    snl_config = SNLConfig(
        mcmc_init_strategy=numpyro.infer.init_to_value(
            values={
                "log_b": np.log(0.26),
                "log_d1": np.log(0.1),
                "log_d2": jnp.log(0.01),
                "p1_": 0.13 + 0.01,
                "p2": 0.05,
            }
        ),
        mcmc_num_chains=args.num_chains,
        train_patience=args.es_threshold,
        train_num_epochs=500,
        train_batch_size=512,
    )
    return snl, snl_config


def run_snl(args, snl_object, snl_config, dataset, logger):
    label = args.label if args.label is not None else ""
    rng = jax.random.PRNGKey(hash("PP" + label))
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


def main(args):
    assert args.obs_error in [0.5, 0.7, 0.9]
    make_folder("PP_experiments")
    current_dir = os.path.dirname(__file__)
    label = "_" + args.label if args.label is not None else ""
    filename = os.path.join(current_dir, f"PP_experiments/PP_{args.obs_error}" + label)
    np.random.seed(123456)
    with Logger(filename + ".log") as logger:

        logger.write(
            f""" 
Experiment details:
    kernel length: {args.kernel_length}
    hidden channels: {args.hidden_channels}
    residual blocks: {args.residual_blocks}
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
        if args.obs_error == 0.9:
            fname = "data/PP_09_data.pkl"
        elif args.obs_error == 0.7:
            fname = "data/PP_07_data.pkl"
        else:
            fname = "data/PP_05_data.pkl"

        with open(fname, "rb") as f:
            obs = dill.load(f)

        def init_val_prior(pred_obs, prey_obs, p, size):
            """Sample pred/prey init values conditioning on pred + prey <= 800"""
            out_pred = np.zeros(size)
            out_prey = np.zeros(size)
            for i in range(size):
                while True:
                    proposed_pred = (
                        np.random.negative_binomial(pred_obs + 1, p) + pred_obs
                    )

                    proposed_prey = (
                        np.random.negative_binomial(prey_obs + 1, p) + prey_obs
                    )
                    if proposed_pred + proposed_prey <= 800:
                        break
                out_pred[i] = proposed_pred
                out_prey[i] = proposed_prey
            return out_pred, out_prey

        def simulate(log_b, log_d1, log_d2, p1, p2):

            if args.obs_error == 0.9:
                init_pred, init_prey = init_val_prior(
                    obs.data[0, 0, 0], obs.data[0, 0, 1], 0.9, size=log_b.shape[0]
                )
            elif args.obs_error == 0.7:
                init_pred, init_prey = init_val_prior(
                    obs.data[0, 0, 0], obs.data[0, 0, 1], 0.7, size=log_b.shape[0]
                )
            else:
                init_pred, init_prey = init_val_prior(
                    obs.data[0, 0, 0], obs.data[0, 0, 1], 0.5, size=log_b.shape[0]
                )

            pred, prey = PP(
                b=np.exp(log_b).reshape(-1),
                d1=np.exp(log_d1).reshape(-1),
                d2=np.exp(log_d2).reshape(-1),
                p1=p1.reshape(-1),
                p2=p2.reshape(-1),
                carrying_capacity=800,
                initial_pred=init_pred,
                initial_prey=init_prey,
                time_run=200,
            )
            if args.obs_error == 0.9:
                pred_noise = np.random.binomial(pred[:, ::2, :].astype(int), p=0.9)
                prey_noise = np.random.binomial(prey[:, ::2, :].astype(int), p=0.9)
            elif args.obs_error == 0.7:
                pred_noise = np.random.binomial(pred[:, ::2, :].astype(int), p=0.7)
                prey_noise = np.random.binomial(prey[:, ::2, :].astype(int), p=0.7)
            else:
                pred_noise = np.random.binomial(pred[:, ::2, :].astype(int), p=0.5)
                prey_noise = np.random.binomial(prey[:, ::2, :].astype(int), p=0.5)

            # Condition on intial observation
            pred_noise[:, 0, :] = obs.data[0, 0, 0]
            prey_noise[:, 0, :] = obs.data[0, 0, 1]

            data = np.concatenate((pred_noise, prey_noise), axis=-1)

            params = np.stack((log_b, log_d1, log_d2, p1, p2), axis=-1)
            return Dataset(
                data=data,
                context=Context(
                    params=params,
                    mask=np.ones_like(data),
                    events=None,
                    left_support=0,
                    right_support=800,
                ),
            )

        log_bs = np.log(0.25) + 0.25 * np.random.normal(size=args.init_samples)
        log_d1s = np.log(0.1) + 0.5 * np.random.normal(size=args.init_samples)
        log_d2s = np.log(0.01) + 0.5 * np.random.normal(size=args.init_samples)
        p1s = np.random.exponential(0.1, size=args.init_samples) + 0.01
        p2s = np.random.exponential(0.05, size=args.init_samples)
        dataset = simulate(log_bs, log_d1s, log_d2s, p1s, p2s)
        model = create_likelihood_model(args)
        snl, snl_config = create_snl(args, model, obs, numpyro_PP, simulate)
        snl, run_time = run_snl(args, snl, snl_config, dataset, logger)

        logger.write(f"\n\nTotal run time: {run_time:.4f} seconds.")

    snl.save_samples(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNL on SIR model")
    parser.add_argument("--kernel-length", default=5, type=int, help="kernel length")
    parser.add_argument(
        "--obs-error",
        default=0.9,
        type=float,
        help="Binomial error on observations",
    )
    parser.add_argument(
        "--num-rounds", default=10, type=int, help="Number of snl rounds"
    )
    parser.add_argument("--label", default=None, type=str, help="experiment label")
    parser.add_argument(
        "--hidden-channels", default=100, type=int, help="num hidden channels"
    )
    parser.add_argument(
        "--residual-blocks", default=3, type=int, help="num hidden layers"
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
        "--max-dataset-size", default=25000, type=int, help="simulation budget"
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
