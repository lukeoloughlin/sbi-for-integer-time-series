import dataclasses
import functools
from typing import Any, Optional, Union, Iterable, Callable
import warnings
import dill
import time

import jax
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value

from .autoregressive_utils import Config
from .training import DistributionModel, TrainState, Dataset
from .sbi_utils import Inference, merge_datasets, train_val_split, to_numpyro

init_to_prior = functools.partial(init_to_median, num_samples=1)

PRNGKey = Array = NumpyroModel = Any


class SNLConfig(Config):
    mcmc_kernel: numpyro.infer.mcmc.MCMCKernel = NUTS
    mcmc_init_strategy: Callable = init_to_prior
    mcmc_num_chains: Union[int, Iterable[int]] = 1
    mcmc_num_warmup_samples: Union[int, Iterable[int]] = 200
    train_batch_size: Union[int, Iterable[int]] = 256
    train_num_epochs: Union[int, Iterable[int]] = 200
    train_val_prop: Union[float, Iterable[float]] = 0.1
    train_patience: Union[int, Iterable[int]] = 20
    train_save_state_on_best_validation: bool = True

    def __getitem__(self, idx):
        return SNLConfig(
            mcmc_kernel=self.mcmc_kernel,
            mcmc_init_strategy=self.mcmc_init_strategy,
            mcmc_num_chains=(
                self.mcmc_num_chains[idx]
                if isinstance(self.mcmc_num_chains, Iterable)
                else self.mcmc_num_chains
            ),
            mcmc_num_warmup_samples=(
                self.mcmc_num_warmup_samples[idx]
                if isinstance(self.mcmc_num_warmup_samples, Iterable)
                else self.mcmc_num_warmup_samples
            ),
            train_batch_size=(
                self.train_batch_size[idx]
                if isinstance(self.train_batch_size, Iterable)
                else self.train_batch_size
            ),
            train_num_epochs=(
                self.train_num_epochs[idx]
                if isinstance(self.train_num_epochs, Iterable)
                else self.train_num_epochs
            ),
            train_val_prop=(
                self.train_val_prop[idx]
                if isinstance(self.train_val_prop, Iterable)
                else self.train_val_prop
            ),
            train_patience=(
                self.train_patience[idx]
                if isinstance(self.train_patience, Iterable)
                else self.train_patience
            ),
            train_save_state_on_best_validation=self.train_save_state_on_best_validation,
        )


def _simulate_data(
    config: SNLConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    simulator: Callable,
    simulator_params: Optional[dict] = None,
    simulator_kwargs: dict = {},
    max_dataset_size: Optional[int] = None,
    _num_simulations: Optional[int] = None,
):
    if simulator_params is not None:
        num_simulations = _num_simulations
        print("Simulating data. ", end="")
        new_dataset = simulator(
            simulator_params, num_samples=num_simulations, **simulator_kwargs
        )
        print("Done.\n")
        new_train, new_val = train_val_split(
            new_dataset, validation_prop=config.train_val_prop
        )
        train_dataset = merge_datasets(train_dataset, new_train, max_dataset_size)
        val_dataset = merge_datasets(val_dataset, new_val, max_dataset_size)

    return train_dataset, val_dataset


def _train_model(
    rng_key: PRNGKey,
    config: SNLConfig,
    likelihood_model: DistributionModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    i: int,
):
    print("Training likelihood model.")
    if isinstance(config.train_patience, int):
        es = config.train_patience
    else:
        es = config.train_patience[i]
    metrics = likelihood_model.train(
        rng_key,
        train_ds=train_dataset,
        validation_ds=val_dataset,
        batch_size=config.train_batch_size,
        num_epochs=config.train_num_epochs,
        save_on_best_val=config.train_save_state_on_best_validation,
        early_stopping_threshold=es,
        verbosity="partial",
    )

    new_state = jax.device_get(likelihood_model.state)
    return metrics, new_state


def _run_mcmc(
    rng_key: PRNGKey,
    config: SNLConfig,
    likelihood_model: DistributionModel,
    numpyro_model: Callable,
    observation: Array,
    num_mcmc_samples: int,
    inference_kwargs: dict = {},
    prev_samples=None,
):
    event_shape = (
        observation.data[0].shape
        if isinstance(observation, Dataset)
        else observation[0].shape
    )
    surrogate_likelihood = to_numpyro(
        likelihood_model.detach_methods(),
        likelihood_model.state,
        event_shape=event_shape,
    )
    if len(prev_samples) == 0:
        init_strategy = config.mcmc_init_strategy
    else:
        last_round = prev_samples[-1]
        values = {key: val[-1] for key, val in last_round.items()}
        init_strategy = init_to_value(values=values)

    kernel = config.mcmc_kernel(numpyro_model, init_strategy=init_strategy)
    num_samples_per_chain = num_mcmc_samples // config.mcmc_num_chains
    print("Sampling with MCMC.\n")
    # Ignore numpyro warning about multiple chains
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mcmc = MCMC(
            kernel,
            num_warmup=config.mcmc_num_warmup_samples,
            num_samples=num_samples_per_chain,
            num_chains=config.mcmc_num_chains,
        )
        mcmc.run(
            rng_key,
            likelihood=surrogate_likelihood,
            obs=observation,
            **inference_kwargs,
        )

    mcmc.print_summary(exclude_deterministic=False)
    samples = jax.device_get(mcmc.get_samples())
    return samples


class SNL(Inference):

    def __init__(
        self,
        distribution_model,
        numpyro_model,
        observation,
        simulator,
        obs_name="obs",
        param_names=None,
        max_dataset_size=80000,
    ):
        super().__init__(observation, simulator, obs_name, param_names)
        self._model = distribution_model
        self._numpyro_model = numpyro_model
        self._max_dataset_size = max_dataset_size
        self._train_metrics = []
        self._train_dataset = None
        self._val_dataset = None

    def to_state_dict(self):
        state_dict = super().to_state_dict()
        state_dict["model"] = self._model.to_state_dict()
        state_dict["numpyro_model"] = self._numpyro_model
        state_dict["max_dataset_size"] = self._max_dataset_size
        state_dict["train_metrics"] = self._train_metrics
        state_dict["train_dataset"] = self._train_dataset
        state_dict["val_dataset"] = self._val_dataset
        state_dict["diagnostics"] = None
        state_dict["parameter"] = None
        state_dict["posterior_metrics"] = None
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict, simulator=None):
        assert state_dict["simulator"] is not None or simulator is not None
        simulator = (
            state_dict["simulator"]
            if state_dict["simulator"] is not None
            else simulator
        )
        distribution_model = DistributionModel.from_state_dict(state_dict["model"])
        snl = cls(
            distribution_model,
            state_dict["numpyro_model"],
            state_dict["observation"],
            simulator,
            state_dict["obs_name"],
            state_dict["param_names"],
            state_dict["max_dataset_size"],
        )
        snl._train_metrics = state_dict["train_metrics"]
        snl._train_dataset = (
            state_dict["train_dataset"]
            if "train_dataset" in state_dict.keys()
            else None
        )
        snl._val_dataset = (
            state_dict["val_dataset"] if "val_dataset" in state_dict.keys() else None
        )
        snl._samples = state_dict["samples"]
        return snl

    def save_samples(self, filename):
        out = {
            "samples": self._samples,
            "obs": self._obs,
            "weights": self._model.state.params,
            "training dataset": self._train_dataset,
            "validation_dataset": self._val_dataset,
            "train_metrics": self._train_metrics,
        }

        with open(filename + ".pkl", "wb") as f:
            dill.dump(out, f, recurse=True)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def state(self):
        return self._model.state

    @property
    def train_metrics(self):
        return self._train_metrics

    @property
    def model(self):
        return self._model

    def set_state(self, state):
        self._model.set_state(state)

    def set_train_dataset(self, dataset):
        self._train_dataset = dataset

    def set_val_dataset(self, dataset):
        self._val_dataset = dataset

    def step(
        self,
        i,
        rng_key,
        config,
        train_dataset,
        val_dataset,
        num_mcmc_samples,
        simulator_kwargs={},
        inference_kwargs={},
        train_model=True,
        run_mcmc=True,
        simulate_data=True,
        logger=None,
    ):
        if logger is not None:
            logger.increment_round()
        sim_time = train_time = mcmc_time = 0
        if simulate_data:
            if logger is not None:
                logger.write(f"Simulating {num_mcmc_samples} samples from simulator.")
                sim_time_0 = time.time()

            simulator_samples = self._samples[-1] if self._samples != [] else None
            train_dataset, val_dataset = _simulate_data(
                config,
                train_dataset,
                val_dataset,
                self.simulator,
                simulator_params=simulator_samples,
                simulator_kwargs=simulator_kwargs,
                _num_simulations=num_mcmc_samples,
            )
            if logger is not None:
                sim_time = time.time() - sim_time_0
                logger.write(f" Finished in {sim_time:.4f} seconds.\n")

        if train_model:
            train_rng, rng_key = jax.random.split(rng_key)
            if logger is not None:
                logger.write(
                    f"Training likelihood model with a dataset size of {train_dataset.data.shape[0]}."
                )
                train_time_0 = time.time()
            metrics, new_state = _train_model(
                train_rng, config, self._model, train_dataset, val_dataset, i
            )
            if logger is not None:
                train_time = time.time() - train_time_0
                logger.write(f" Finished in {train_time:.4f} seconds.\n")
            self._model.set_state(new_state)

        if run_mcmc:
            mcmc_rng, rng_key = jax.random.split(rng_key)
            if logger is not None:
                logger.write(
                    f"""Sampling {num_mcmc_samples} posterior samples with MCMC over {config.mcmc_num_chains} chains."""
                )
                mcmc_time_0 = time.time()
            samples = _run_mcmc(
                mcmc_rng,
                config,
                self._model,
                self._numpyro_model,
                self.obs,
                num_mcmc_samples,
                inference_kwargs,
                self._samples,
            )
            if logger is not None:
                mcmc_time = time.time() - mcmc_time_0
                logger.write(f" Finished sampling in {mcmc_time:.4f} seconds.\n")

        self._samples.append(samples)
        self._train_metrics.append(metrics)
        self.set_train_dataset(train_dataset)
        self.set_val_dataset(val_dataset)

    def SNL(
        self,
        rng_key,
        config,
        dataset=None,
        mcmc_samples_per_round=None,
        simulator_kwargs={},
        inference_kwargs={},
        max_num_rounds=10,
        logger=None,
    ):
        if dataset is None:
            assert self.train_dataset is not None, "no dataset is currently set"
            dataset = self.train_dataset
        else:
            train_dataset, val_dataset = train_val_split(dataset, config.train_val_prop)
            self.set_train_dataset(train_dataset)
            self.set_val_dataset(val_dataset)

        if mcmc_samples_per_round is None:
            num_samples_per_round = round(
                (self._max_dataset_size - self.train_dataset.data.shape[0])
                / max_num_rounds
            )
        else:
            num_samples_per_round = mcmc_samples_per_round

        for i in range(max_num_rounds):
            if logger is not None:
                round_time_0 = time.time()
            snl_rng, rng_key = jax.random.split(rng_key, num=2)
            self.step(
                i,
                snl_rng,
                config,
                self.train_dataset,
                self.val_dataset,
                simulator_kwargs=simulator_kwargs,
                inference_kwargs=inference_kwargs,
                train_model=True,
                run_mcmc=True,
                simulate_data=(i != 0),
                num_mcmc_samples=num_samples_per_round,
                logger=logger,
            )
            self._train_dataset = jax.device_get(self._train_dataset)
            self._val_dataset = jax.device_get(self._val_dataset)

            if logger is not None:
                round_time = time.time() - round_time_0
                logger.write(
                    f"Finsihed running round {logger.round} of SNL in {round_time:.4f} seconds.\n"
                )

    def mcmc(
        self,
        kernel,
        num_samples,
        num_chains=1,
        num_warmup=200,
        init_strategy=init_to_prior(),
        **inference_kwargs,
    ):
        surrogate_likelihood = to_numpyro(
            self._model.detach_methods(), self.state, event_shape=self.obs[0].shape
        )
        # Sample the posterior with MCMC
        mcmc_rng, rng_key = jax.random.split(rng_key)
        kernel = kernel(self.numpyro_model, init_strategy=init_strategy)
        num_samples_per_chain = num_samples // num_chains
        print("Sampling with MCMC.\n")
        # Ignore numpyro warning about multiple chains
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples_per_chain,
                num_chains=num_chains,
            )
            mcmc.run(
                mcmc_rng,
                likelihood=surrogate_likelihood,
                obs=self.obs,
                **inference_kwargs,
            )

        mcmc.print_summary()
        return jax.device_get(mcmc.get_samples())
