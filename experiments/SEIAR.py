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


from experiments.simulators import SEIAR


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


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


def logit(p):
    return np.log(p) - np.log(1 - p)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def create_likelihood_model(args):
    def event_mask_fn(x, *_args):
        mask = jnp.ones_like(x)
        midpoints_mask = (x[:, 1:, :] - x[:, :-1, :]) != 0
        mask = mask.at[:, 1:, :].set(midpoints_mask)
        mask = mask * (x != args.pop_size)
        return mask

    network_config = NetworkConfig(
        kernel_shape=args.kernel_length,
        hidden_channels=args.hidden_channels,
        residual_blocks=args.residual_blocks,
        data_scale=np.min([args.pop_size, 500]),
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
        monotonic=True,
        mixture_components=args.mixture_components,
        event_mask_fn=event_mask_fn,
        epidemic=True,
    )

    return DistributionModel(
        model,
        optimizer=optax.adamw,
        optimizer_kwargs={"learning_rate": args.lr, "weight_decay": args.weight_decay},
    )


def numpyro_SEIAR(likelihood, obs=None):
    R0 = numpyro.sample("R0", dist.Uniform(0.1, 8))
    sigma_inv = numpyro.sample("sigma_inv", TruncatedGamma(10, 10, 0.1))
    gamma_inv = numpyro.sample("gamma_inv", TruncatedGamma(10, 10, 0.5))
    kappa = numpyro.sample("kappa", dist.Uniform(0, 1))
    q = numpyro.sample("q", dist.Uniform(0.5, 1))

    numpyro.sample(
        "no_infection", dist.Bernoulli(1 - q), obs=0
    )  # initial infection becomes asymptomatic

    logit_kappa = jnp.log(kappa) - jnp.log(1 - kappa)
    logit_q = jnp.log(q) - jnp.log(1 - q)

    params = stack_and_broadcast(
        R0 * jnp.ones(1),
        sigma_inv * jnp.ones(1),
        gamma_inv * jnp.ones(1),
        logit_kappa * jnp.ones(1),
        logit_q * jnp.ones(1),
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


##########################################################################################


def create_snl(args, model, observation, numpyro_model, simulator):
    snl = SNL(
        model,
        numpyro_model,
        observation,
        simulator,
        param_names=["R0", "sigma_inv", "gamma_inv", "kappa", "q"],
        max_dataset_size=args.max_dataset_size,
    )
    snl_config = SNLConfig(
        mcmc_init_strategy=numpyro.infer.init_to_value(
            values={
                "R0": 2.2,
                "gamma_inv": 1.0,
                "sigma_inv": 1.0,
                "kappa": 0.7,
                "q": 0.9,
            }
        ),
        mcmc_num_chains=args.num_chains,
        train_patience=args.es_threshold,
        train_num_epochs=500,
        train_batch_size=256,
    )
    return snl, snl_config


def run_snl(args, snl_object, snl_config, dataset, logger):
    label = args.label if args.label is not None else ""
    rng = jax.random.PRNGKey(hash("SEIAR" + label))
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


def create_mask_and_events(infection_plus_exposure, Z3):
    die_out = infection_plus_exposure[:, -1, 0] == 0
    mask = np.ones_like(Z3)
    events = -np.ones((infection_plus_exposure.shape[0], 1))
    for i in range(infection_plus_exposure.shape[0]):
        if die_out[i]:
            mask_idx = np.where(Z3[i, :, 0] == Z3[i, -1, 0])[0][0]
            if mask_idx < (mask.shape[1] - 1):
                mask[i, (mask_idx + 1) :, 0] = 0.0
            events[i, 0] = mask_idx
    return mask, events.astype(int)


def main(args):
    assert args.pop_size in [150, 350, 500, 1000, 2000]
    make_folder("SEIAR_experiments")
    current_dir = os.path.dirname(__file__)
    label = "_" + args.label if args.label is not None else ""
    filename = os.path.join(
        current_dir, f"SEIAR_experiments/SEIAR_{args.pop_size}_obs" + label
    )
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

        with open(f"data/SEIAR_{args.pop_size}_pop_data.pkl", "rb") as f:
            obs = dill.load(f)

        if args.pop_size == 150:
            tr = 70  # 0.085% don't fade out
        elif args.pop_size == 350:
            tr = 70  # 0.02% don't fade out
        elif args.pop_size == 500:
            tr = 50  # 0.027% don't fade out
        else:
            tr = 60  # 0.02% don't fade out

        def simulate(R0, sigma_inv, gamma_inv, kappa, q):
            sigma = 1 / sigma_inv
            gamma = 1 / gamma_inv
            beta_p = gamma * kappa * R0 / q
            beta_s = gamma * (1 - kappa) * R0 / q
            Z1, Z2, Z3, Z4, Z5 = SEIAR(
                N=args.pop_size,
                time_run=tr,
                beta_p=beta_p,
                beta_s=beta_s,
                sigma=sigma,
                gamma=gamma,
                q=q,
                initial_infected=1,
            )
            infected_plus_exposed = (Z1 - Z2 - Z5) + (Z2 - Z4)
            mask, events = create_mask_and_events(infected_plus_exposed, Z3)
            params = np.stack(
                (R0, sigma_inv, gamma_inv, logit(kappa), logit(q)), axis=-1
            )
            return Dataset(
                data=Z3,
                context=Context(
                    params=params,
                    mask=mask,
                    events=events,
                    left_support=0,
                    right_support=args.pop_size,
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

        R0s = 0.1 + 7.9 * np.random.rand(args.init_samples)
        sigma_invs = truncated_gamma(10, 0.1, 0.1, size=args.init_samples)
        gamma_invs = truncated_gamma(10, 0.1, 0.5, size=args.init_samples)
        kappas = np.random.rand(args.init_samples)
        qs = 0.5 + 0.5 * np.random.rand(args.init_samples)
        dataset = simulate(R0s, sigma_invs, gamma_invs, kappas, qs)
        model = create_likelihood_model(args)
        snl, snl_config = create_snl(args, model, obs, numpyro_SEIAR, simulate)
        snl, run_time = run_snl(args, snl, snl_config, dataset, logger)

        logger.write(f"\n\nTotal run time: {run_time:.4f} seconds.")

    snl.save_samples(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNL on SEIAR model")
    parser.add_argument("--kernel-length", default=5, type=int, help="kernel length")
    parser.add_argument(
        "--pop-size",
        default=350,
        type=int,
        help="number of trajectories to use as observed data",
    )
    parser.add_argument(
        "--num-rounds", default=10, type=int, help="Number of snl rounds"
    )
    parser.add_argument("--label", default=None, type=str, help="experiment label")
    parser.add_argument(
        "--hidden-channels", default=64, type=int, help="num hidden channels"
    )
    parser.add_argument(
        "--residual-blocks", default=3, type=int, help="num hidden layers"
    )
    parser.add_argument(
        "--param-encode-dim", default=12, type=int, help="encoding dim of parameters"
    )  # 50
    parser.add_argument(
        "--param-hidden-dim",
        default=64,
        type=int,
        help="hidden dim of parameter encoder",
    )  # 256
    parser.add_argument(
        "--mixture-components",
        default=5,
        type=int,
        help="number of logistic mixture components",
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--weight-decay", default=1e-5, type=float, help="weight decay")
    parser.add_argument(
        "--max-dataset-size", default=60000, type=int, help="simulation budget"
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
