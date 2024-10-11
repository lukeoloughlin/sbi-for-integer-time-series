import copy
from collections import OrderedDict
import functools
import inspect
from typing import Callable, Any, Optional, NamedTuple
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import tqdm.notebook as tqdmn
from flax.training import train_state
import flax.linen as nn

PRNGKey = Array = Optimizer = Any


class Dataset(NamedTuple):
    data: Array
    context: Optional[Array] = None

    def __getitem__(self, idx):
        return Dataset(self.data[idx], self.context[idx])


class TrainState(train_state.TrainState):
    conditional: bool = False


def to_pickleable_config(config):
    try:
        config_type = type(config)
        config = config._asdict()
        return config_type(**config)
    except:
        return config


def picklable_model(model):
    model_type = type(model)
    model_dict = dataclasses.asdict(model)
    for key, value in model_dict.items():
        if isinstance(value, tuple):
            model_dict[key] = to_pickleable_config(value)
    return model_type(**model_dict)


def autoregressive_init(
    autoregressive_model: nn.Module,
    rng: PRNGKey = None,
    optimizer: Optimizer = None,
    dataset: Dataset = None,
    num_init_samples=10,
    state_dict: dict = None,
):
    if state_dict is not None:
        del rng, optimizer, dataset, num_init_samples
        train_state = TrainState.create(
            apply_fn=autoregressive_model.apply,
            params=state_dict["state"]["params"],
            tx=state_dict["optimizer_function"](**state_dict["optimizer_kwargs"]),
            batch_stats=state_dict["state"]["batch_stats"],
            conditional=state_dict["state"]["conditional"],
        )
        train_state = dataclasses.asdict(train_state)
        train_state["opt_state"] = state_dict["state"]["opt_state"]
        train_state = TrainState(**train_state)
        has_ctx = train_state.conditional
    else:
        init_input = dataset.data[:num_init_samples]
        init_context = (
            dataset.context[:num_init_samples] if dataset.context is not None else None
        )
        init_state = autoregressive_model.init(rng, init_input, context=init_context)
        params = init_state["params"]
        has_ctx = init_context is not None
        train_state = TrainState.create(
            apply_fn=autoregressive_model.apply,
            params=params,
            tx=optimizer,
            conditional=has_ctx,
        )

    logpdf = _compile_logpdf(autoregressive_model, has_ctx)
    param_fn = _compile_param_fn(autoregressive_model, has_ctx)
    return train_state, logpdf, param_fn


def _compile_logpdf(autoregressive_model, has_ctx):
    def logpdf(params, inputs, context):
        return autoregressive_model.apply({"params": params}, inputs, context=context)

    if not has_ctx:
        logpdf = functools.partial(logpdf, context=None)
    return jax.jit(logpdf)


def _compile_param_fn(autoregressive_model, has_ctx):
    def autoregressive_params(params, inputs, context):
        return autoregressive_model.apply(
            {"params": params}, inputs, context=context, _method="params"
        )

    if not has_ctx:
        autoregressive_params = functools.partial(autoregressive_params, context=None)

    return jax.jit(autoregressive_params)


def _check_signature_string(fn: Callable, arg: str, arg_value: str):
    """Check whether an argument has been partially applied"""
    return str(inspect.signature(fn).parameters[arg]) == (arg + "=" + arg_value)


def train_step(logpdf_fn):
    has_ctx = not _check_signature_string(logpdf_fn, "context", "None")

    @jax.jit
    def evaluate_train_step(state, data, context):
        def loss_fn(params):
            loglikelihood = logpdf_fn(params=params, inputs=data, context=context)
            return -jnp.mean(loglikelihood)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, -loss

    if not has_ctx:
        evaluate_train_step = functools.partial(evaluate_train_step, context=None)
    return evaluate_train_step


def eval_step(logpdf_fn):
    has_ctx = not _check_signature_string(logpdf_fn, "context", "None")

    @jax.jit
    def evaluate_eval_step(state, data, context):
        eval_logpdf = jnp.mean(
            logpdf_fn(params=state.params, inputs=data, context=context)
        )
        return eval_logpdf

    if not has_ctx:
        evaluate_eval_step = functools.partial(evaluate_eval_step, context=None)
    return evaluate_eval_step


def train_epoch(logpdf_fn, running_average_momentum=0.9):
    train_step_ = train_step(logpdf_fn)
    has_ctx = not _check_signature_string(logpdf_fn, "context", "None")

    @jax.jit
    def train_epoch_single_perm(i, val):
        perms = val[-1]
        perm = perms[i]
        if has_ctx:
            state, data, context, running_logpdf, _ = val
            state, logpdf = train_step_(state, data[perm], context[perm])
            running_logpdf = running_average_momentum * running_logpdf + (
                1 - running_average_momentum
            ) * jnp.mean(logpdf)
            return (state, data, context, running_logpdf, perms)
        else:
            state, data, running_logpdf, _ = val
            state, logpdf = train_step_(state, data[perm])
            running_logpdf = running_average_momentum * running_logpdf + (
                1 - running_average_momentum
            ) * jnp.mean(logpdf)
            return (state, data, running_logpdf, perms)

    def evaluate_train_epoch(state, train_ds, batch_size, rng):
        assert isinstance(
            train_ds, Dataset
        ), "Dataset should be organised using the Dataset class"
        rng, perm_rng = jax.random.split(rng)
        train_ds_size = train_ds.data.shape[0]
        steps_per_epoch = train_ds_size // batch_size
        perms = jax.random.permutation(perm_rng, train_ds_size)
        # skip incomplete batch
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        if has_ctx:
            init_val = (state, train_ds.data, train_ds.context, 0, perms)
        else:
            init_val = (state, train_ds.data, 0, perms)
        out = jax.lax.fori_loop(0, steps_per_epoch, train_epoch_single_perm, init_val)
        batch_logpdf = jax.device_get(out[-2])
        state = out[0]
        return state, batch_logpdf

    return evaluate_train_epoch


def validation_epoch(logpdf_fn, max_validation_size=10000):
    eval_step_ = eval_step(logpdf_fn)
    has_ctx = not _check_signature_string(logpdf_fn, "context", "None")

    def evaluate_validation_epoch(state, test_ds):
        assert isinstance(
            test_ds, Dataset
        ), "Dataset should be organised using the Dataset class"
        num_splits = np.max(np.array([test_ds.data.shape[0] // max_validation_size, 1]))
        datas = jnp.array_split(test_ds.data, num_splits)
        weights = [data.shape[0] / test_ds.data.shape[0] for data in datas]
        eval_logpdf = 0
        if has_ctx:
            batch_sizes = [0, *[batch.shape[0] for batch in datas]]
            indices = np.cumsum(np.array(batch_sizes))
            ctxs = [
                test_ds.context[top_idx:bottom_idx]
                for top_idx, bottom_idx in zip(indices[:-1], indices[1:])
            ]
            for data, ctx, weight in zip(datas, ctxs, weights):
                eval_logpdf = eval_logpdf + weight * eval_step_(
                    state, data=data, context=ctx
                )
        else:
            for data, weight in zip(datas, weights):
                eval_logpdf = eval_logpdf + weight * eval_step_(state, data=data)
        return jax.device_get(eval_logpdf)

    return evaluate_validation_epoch


def train_loop(
    rng,
    state,
    train_epoch_fn,
    validation_epoch_fn,
    train_ds,
    validation_ds,
    batch_size,
    num_epochs,
    save_on_best_val=True,
    early_stopping_threshold=None,
    notebook_format=False,
    verbosity="full",
):
    # Check which progress bar to use
    if verbosity == "full":
        tqdm_ = tqdmn.tqdm if notebook_format else tqdm.tqdm
    elif verbosity == "partial":
        tqdm_ = tqdm.tqdm
    current_best_val_logpdf = -np.inf
    if early_stopping_threshold is not None:
        assert isinstance(
            early_stopping_threshold, int
        ), "early stopping threshold must be an integer"
        epochs_elapsed = 0
    if save_on_best_val:
        return_state = copy.deepcopy(jax.device_get(state))
    # Setup progress bars
    if verbosity is not None:
        progress = tqdm_(total=num_epochs, desc="Progress", position=0)
        if verbosity == "full":
            train_progress = tqdm_(total=0, position=1, bar_format="{desc}")
            val_progress = tqdm_(total=0, position=2, bar_format="{desc}")
    if save_on_best_val and verbosity == "full":
        best_progress = tqdm_(total=0, position=3, bar_format="{desc}")
    # Train loop
    metrics = {"train": [], "validation": []}
    for _ in range(num_epochs):
        # Train and validate over a single epoch
        epoch_rng, rng = jax.random.split(rng)
        state, train_logpdf = train_epoch_fn(state, train_ds, batch_size, epoch_rng)
        val_logpdf = validation_epoch_fn(state, validation_ds)
        # Update metrics
        metrics["train"].append(train_logpdf)
        metrics["validation"].append(val_logpdf)
        # Check if updated state is the best so far
        if val_logpdf > current_best_val_logpdf:
            current_best_val_logpdf = val_logpdf
            if early_stopping_threshold is not None:
                epochs_elapsed = 0
            if save_on_best_val:
                return_state = copy.deepcopy(jax.device_get(state))
        else:
            if early_stopping_threshold is not None:
                epochs_elapsed += 1
        # Update progress bar(s)
        progress.update(1)
        if verbosity == "partial":
            msg = OrderedDict()
            msg["validation log pdf"] = f"{val_logpdf:.4f}"
            progress.set_postfix(ordered_dict=msg)
        if verbosity == "full":
            train_progress.set_description_str(
                f"Current epoch train log pdf: {train_logpdf:.4f}"
            )
            val_progress.set_description_str(
                f"Current validation epoch log pdf: {val_logpdf:.4f}"
            )
            if save_on_best_val:
                best_progress.set_description_str(
                    f"Best validation log pdf: {current_best_val_logpdf:.4f}"
                )
        # Check whether to activate early stopping
        if early_stopping_threshold is not None:
            if epochs_elapsed == early_stopping_threshold:
                print(
                    f"Early stopping activated with no improvement in validation log pdf after {epochs_elapsed} epochs."
                )
                break
    if save_on_best_val:
        if verbosity == "partial":
            print(
                f"Training finished. Best validation log pdf: {current_best_val_logpdf:.4f}\n"
            )
        return return_state, metrics
    else:
        return state, metrics


class AutoregressiveContainer(NamedTuple):
    """Hold the compiled logpdf function of the autoregressive model for use in numpyro"""

    logpdf: Callable
    params: Callable


class DistributionModel(object):
    """Holds a distribution model, managing the model state and implementing training."""

    def __init__(
        self,
        model,
        optimizer,
        optimizer_kwargs={},
        state=None,
        rng=None,
        dataset=None,
        num_init_samples=10,
        train_loss_momentum=0.9,
        max_batch_size=1000,
    ):
        self._model = model
        self._optimizer = optimizer(**optimizer_kwargs)
        self._optim_fn = optimizer
        self._optim_kwargs = optimizer_kwargs
        self._train_loss_momentum = train_loss_momentum
        self._max_batch_size = max_batch_size
        self._state = state
        if dataset is None:
            self.is_initialised = False
        else:
            self.init(rng, dataset, num_init_samples, state=state)

    def to_state_dict(self):
        assert (
            self.is_initialised
        ), "Cannot create state_dict since model parameters have not been initilialised."
        state_dict = {
            "model": picklable_model(self._model),
            "optimizer_function": self._optim_fn,
            "optimizer_kwargs": self._optim_kwargs,
            "train_loss_momentum": self._train_loss_momentum,
            "max_batch_size": self._max_batch_size,
            "state": {
                "params": self._state.params,
                "opt_state": self._state.opt_state,
                "conditional": self._state.conditional,
            },
        }
        return state_dict

    def init(self, rng, dataset, num_init_samples=10, state=None):
        _state, logpdf, param_fn = autoregressive_init(
            self._model,
            rng,
            self._optimizer,
            dataset,
            num_init_samples=num_init_samples,
        )
        if state is None:
            self._state = _state
        self._logpdf = logpdf
        self._param_fn = param_fn

        self._train_epoch = train_epoch(self._logpdf, self._train_loss_momentum)
        self._validation_epoch = validation_epoch(self._logpdf, self._max_batch_size)
        self.is_initialised = True

    def train(
        self,
        rng,
        train_ds,
        validation_ds,
        batch_size,
        num_epochs,
        save_on_best_val=True,
        early_stopping_threshold=None,
        verbosity="full",
        num_init_samples=10,
    ):
        if not self.is_initialised:
            init_rng, rng = jax.random.split(rng)
            self.init(init_rng, train_ds, num_init_samples)
        try:
            if verbosity == "full":
                get_ipython()
                notebook_format = True
            else:
                notebook_format = False
        except:
            notebook_format = False
        updated_state, metrics = train_loop(
            rng,
            self._state,
            self._train_epoch,
            self._validation_epoch,
            train_ds,
            validation_ds,
            batch_size,
            num_epochs,
            save_on_best_val=save_on_best_val,
            early_stopping_threshold=early_stopping_threshold,
            notebook_format=notebook_format,
            verbosity=verbosity,
        )
        self._state = updated_state
        return metrics

    def logpdf(self, inputs, context=None):
        if self.is_initialised:
            return self._logpdf(self.state.params, inputs=inputs, context=context)
        else:
            raise Exception("cannot call logpdf until model is initialised")

    def params(self, inputs, context=None):
        return self._param_fn(self.state.params, inputs, context=context)

    def param_gradients(self, inputs, context=None):
        if self.is_initialised:

            def loss_fn(params):
                loglikelihood = self._logpdf(params, inputs=inputs, context=context)
                return -jnp.mean(loglikelihood)

            grad_fn = jax.value_and_grad(loss_fn)
            return grad_fn(self.state.params)
        else:
            raise Exception("Cannot call param_gradients until model is initialised")

    @property
    def state(self):
        return self._state

    def set_state(self, new_state):
        """Set the train state"""
        self._state = new_state

    def set_optimizer(self, new_optim):
        self._optimizer = new_optim

    def detach_methods(self, method="all"):
        """Extract the pure functions associated with the model."""
        if method == "logpdf":
            return functools.partial(self._logpdf, validate=False)
        elif method == "params":
            return self._param_fn
        elif method == "all":
            return AutoregressiveContainer(logpdf=self._logpdf, params=self._param_fn)

    def device_get(self):
        if self._state is not None:
            self._state = jax.device_get(self._state)

    def device_put(self):
        if self._state is not None:
            self._state = jax.device_put(self._state)
