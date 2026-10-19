### Lena Dogra, October 2026
### functions for QML models based on parameterized circuits in circuits.py

import csv
import os
import sys
import time
import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import pennylane as qml
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce
import json
import re
import libs.circuits
import libs.dataloading

#### generating and training QML models based on PQC from circuits.py and a local or global observable from Z-measurements
# --- Model, Loss, Optimizer ---
def make_model(dev, circuit, local_observable):
    model_name = str(circuit.n_wires) + 'qubits_' + circuit.circuit_name + str(circuit.n_enc)+'enc_' + str(circuit.n_dec)+'dec_' +('' if local_observable else 'non') + 'local'
    no_gate_angles = circuit.n_var_angles
    no_upload_weights = circuit.n_encoding_angles if circuit.choose_freq else 0
    no_params = no_gate_angles  + no_upload_weights
    if local_observable:
        observable_indices = list(range(circuit.n_wires))
        obs_return = lambda: [qml.expval(qml.PauliZ(i)) for i in observable_indices]
    else:
        full_obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(circuit.n_wires)])
        obs_return = lambda: [qml.expval(full_obs)]
    @qml.qnode(dev, interface="jax")
    def qnn_pqc(inputs, angles):
        circuit(inputs, angles,  dev.wires)
        return obs_return()
    @jax.jit
    def model(params, inputs):
        outputs = qnn_pqc(inputs,params)
        return jnp.squeeze(outputs[0])
    return model, no_params, model_name
# ============================ checkpointing ============================
# A checkpoint = params + optax state (Adam m/v/step) + loss history + the
# settings it was trained with (validated on resume). Written atomically
# (tmp file + os.replace) from the Python epoch loop, so it can never be
# read half-written, and it only touches a few KB per epoch.
CKPT_VERSION = 1

def save_checkpoint(path, params, opt_state, epoch, history, cfg, initial_params):
    """Atomically write a training checkpoint after `epoch` completed epochs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": CKPT_VERSION,
        "cfg": cfg,
        "epoch": int(epoch),
        "initial_params": np.asarray(initial_params),
        "history": jax.tree_util.tree_map(np.asarray, history),
        "params": jax.tree_util.tree_map(np.asarray, params),
        "opt_state": jax.tree_util.tree_map(np.asarray, opt_state),
    }
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)

def load_checkpoint(path, cfg, n_epochs):
    """Return the checkpoint payload if it is compatible with this run, else None.
    Compatible = same model/key/lr/batch/data; n_epochs may have increased
    (shuffles beyond the checkpointed epoch are then re-derived from the same
    key stream -- printed as a NOTE)."""
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"WARNING: unreadable checkpoint {path}: {e}; starting fresh", flush=True)
        return None
    if payload.get("version") != CKPT_VERSION:
        print(f"WARNING: checkpoint {path} has version {payload.get('version')} != {CKPT_VERSION}; starting fresh", flush=True)
        return None
    p = payload.get("cfg", {})
    for k, v in cfg.items():
        if k == "n_epochs":
            continue                              # handled below
        if p.get(k) != v:
            print(f"WARNING: checkpoint {path} cfg mismatch on '{k}' ({p.get(k)!r} != {v!r}); starting fresh", flush=True)
            return None
    p_ne = p.get("n_epochs")
    if p_ne is None or p_ne < payload["epoch"]:
        print(f"WARNING: checkpoint {path} inconsistent (n_epochs={p_ne}, epoch={payload['epoch']}); starting fresh", flush=True)
        return None
    if p_ne > n_epochs:
        print(f"WARNING: checkpoint {path} ran {p_ne} epochs > requested {n_epochs}; starting fresh", flush=True)
        return None
    if p_ne != n_epochs:
        print(f"NOTE: n_epochs increased {p_ne} -> {n_epochs}; per-epoch shuffles from epoch {p_ne} on "
              f"are re-derived from the same key stream (params and optimizer state carry over).", flush=True)
    for field in ("params", "opt_state", "history"):
        payload[field] = jax.tree_util.tree_map(jnp.asarray, payload[field])
    payload["initial_params"] = jnp.asarray(payload["initial_params"])
    return payload
# --- Training ---
def train_model(args_opt, model, params, train_inputs, train_targets, val_inputs, val_targets,
                shuffle_key, opt_state=None, loss_history=None, start_epoch=0,
                epoch_callback=None):
    """Train the model epoch by epoch.

    Identical to the old fully-jitted fori_loop version in everything that
    affects the result (same per-epoch shuffle keys, same init, same
    optimizer, same ops in the same order); the epoch loop itself just runs
    in Python so a checkpoint can be written after each epoch:

      * the per-epoch step is ONE jitted function, compiled once
      * opt_state=None    -> opt.init(params), as before (deterministic, no RNG)
      * loss_history=None -> fresh zero-filled history, as before
      * start_epoch       -> first epoch to run (0 for fresh runs)
      * epoch_callback    -> called as (epochs_done, params, opt_state,
                              loss_history) from Python after each epoch
    """
    opt, loss, no_epochs, batch_size = args_opt
    trainset_size = train_inputs.shape[0]
    valset_size = val_inputs.shape[0]
    no_features = train_inputs.shape[1]
    no_batches_train = int(trainset_size / batch_size)
    no_batches_val = int(valset_size / batch_size)
    # one key per epoch, split up front (constant; only touched when shuffling)
    epoch_keys = jax.random.split(shuffle_key, no_epochs) if shuffle_key is not None else None
    ### compile the loss function
    @jax.jit
    def loss_fn(params, inputs, targets):
        return loss(params, inputs, targets, model=model)
    ### compile the update step for one batch
    @jax.jit
    def update_step_batch_jit(i, args):
        params, opt_state, in_train, out_train = args
        ins = jax.lax.dynamic_slice(in_train, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_train, [i*batch_size], [batch_size])
        loss_val, grads = jax.value_and_grad(loss_fn)(params, ins, outs)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, in_train, out_train)
    ### compile function for calculating loss over one batch
    @jax.jit
    def calculate_dataset_loss_jit(i, args):
        loss_value, params, in_data, out_data = args
        ins = jax.lax.dynamic_slice(in_data, [i*batch_size, 0], [batch_size, no_features])
        outs = jax.lax.dynamic_slice(out_data, [i*batch_size], [batch_size])
        loss_batch = loss_fn(params, ins, outs)
        loss_value = loss_value + loss_batch
        return (loss_value, params, in_data, out_data)
    ### compile the update step for one epoch
    @jax.jit
    def update_step_epoch_jit(i, args):
        params, loss_hist, opt_state, in_train, out_train, in_val, out_val = args
        # fresh JAX shuffle each epoch (statically absent when shuffle_key is None)
        if epoch_keys is not None:
            perm = jax.random.permutation(epoch_keys[i], trainset_size)
            inputs, outputs = in_train[perm], out_train[perm]
        else:
            inputs, outputs = in_train, out_train
        args_batch = (params, opt_state, inputs, outputs)
        (params, opt_state, _, _) = jax.lax.fori_loop(0, no_batches_train, update_step_batch_jit, args_batch)
        # calculate train and validation losses at the end of each epoch
        train_loss = 0.0; args_train_loss = (train_loss, params, in_train, out_train)
        (train_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_train, calculate_dataset_loss_jit, args_train_loss)
        train_loss = train_loss / no_batches_train
        val_loss = 0.0; args_val_loss = (val_loss, params, in_val, out_val)
        (val_loss, _, _, _) = jax.lax.fori_loop(0, no_batches_val, calculate_dataset_loss_jit, args_val_loss)
        val_loss = val_loss / no_batches_val
        # update loss history container
        train_hist = loss_hist['train_loss'].at[i].set(train_loss)
        val_hist = loss_hist['val_loss'].at[i].set(val_loss)
        loss_hist['train_loss'] = train_hist
        loss_hist['val_loss'] = val_hist
        # Print the loss
        jax.debug.print("Epoch {ep}  ----  Train loss: {train_loss}  ----  Val. loss: {val_loss}", ep=(i+1), train_loss=train_loss, val_loss=val_loss)
        return (params, loss_hist, opt_state, in_train, out_train, in_val, out_val)
    ### run the optimization: the epoch step above is compiled once and driven
    ### from Python so a checkpoint can be written after each epoch
    if opt_state is None:
        opt_state = opt.init(params)
    if loss_history is None:
        loss_history = {'train_loss': jnp.zeros(no_epochs),
                        'val_loss': jnp.zeros(no_epochs)}
    for epoch in range(int(start_epoch), no_epochs):
        args = (params, loss_history, opt_state, train_inputs, train_targets,
                val_inputs, val_targets)
        (params, loss_history, opt_state, _, _, _, _) = update_step_epoch_jit(epoch, args)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, params, opt_state, loss_history)
    return params, loss_history
# mean squared error loss function
def mse_loss(params, inputs, targets, model):
    predictions = model(params, inputs)
    loss = jnp.sum((targets - predictions) ** 2.0) / len(targets)
    return loss
# --- Training and evaluation ---
def train_and_evaluate(model, model_name, no_params,
                       key,
                       train_inputs, train_outputs, val_inputs, val_outputs,
                       save_dir,
                       initial_params = 'random_uniform',
                       return_optpars = False,
                       save_postfix = '',
                       loss = mse_loss,
                       plot_hist = True,
                       learning_rate = 0.002, n_epochs = 150, batch_size = 50,
                       write_csv = True,
                       ckpt_path = None, save_ckpt_every = 1, ckpt_tag = ''):
    """As before, plus checkpoint/resume support.

    With ckpt_path=None (or if no compatible checkpoint exists) the run is
    exactly the same as before: same init, same shuffles, same optimizer
    steps. If ckpt_path points at a compatible checkpoint, training
    CONTINUES from it (params + Adam state + epoch counter); a finished
    job's checkpoint is deleted by the caller (train_sweep).
    """
    os.makedirs(save_dir, exist_ok=True)
    # 'key' is used directly for param init; a deterministic separate stream
    # derived from it is used for the per-epoch shuffles
    shuffle_key = jax.random.fold_in(key, 1)
    match = re.match(r"(\d+)qubits.*?(\d+)enc_(\d+)dec_([a-zA-Z]+)", model_name)
    if match:
        n_qubits, n_enc, n_dec, entanglement = match.groups()
        n_qubits, n_enc, n_dec, entanglement = int(n_qubits), int(n_enc), int(n_dec), int(entanglement == 'local')
    # identity + settings a checkpoint must match before it may be resumed
    ckpt_cfg = {
        'model_name': model_name,
        'no_params': int(no_params),
        'key': str(np.asarray(key).ravel().tolist()),
        'n_epochs': int(n_epochs),
        'learning_rate': float(learning_rate),
        'batch_size': int(batch_size),
        'tag': ckpt_tag,
    }
    if (type(initial_params) == str) and (initial_params == 'random_uniform'):
        params = jax.random.uniform(key, shape=(no_params,), minval=-1.0, maxval=1.0)
        if 'freq' in model_name:
            params = params.at[:n_enc*n_qubits].set(params[:n_enc*n_qubits] * 0.01 + 1)
    else:
        params = initial_params
    initial_params_vector = params
    opt = optax.adam(learning_rate=learning_rate)
    # ---- resume from a compatible checkpoint, if any ----
    ckpt = load_checkpoint(ckpt_path, ckpt_cfg, n_epochs) if ckpt_path is not None else None
    if ckpt is not None:
        print(f"RESUME {model_name}: continuing from epoch {ckpt['epoch']}/{n_epochs}", flush=True)
        params = ckpt['params']
        initial_params_vector = ckpt['initial_params']
        opt_state = ckpt['opt_state']
        history = ckpt['history']
        start_epoch = int(ckpt['epoch'])
        # allow n_epochs to have increased since the checkpoint was written
        if history['train_loss'].shape[0] < n_epochs:
            history = {'train_loss': jnp.concatenate([history['train_loss'],
                                                      jnp.zeros(n_epochs - history['train_loss'].shape[0])]),
                       'val_loss': jnp.concatenate([history['val_loss'],
                                                    jnp.zeros(n_epochs - history['val_loss'].shape[0])])}
    else:
        opt_state, history, start_epoch = None, None, 0
    # ---- write a checkpoint after each epoch (Python side, outside the jit) ----
    if ckpt_path is not None:
        def epoch_callback(epoch_done, p, os_, hist):
            if epoch_done % save_ckpt_every == 0 or epoch_done == n_epochs:
                save_checkpoint(ckpt_path, p, os_, epoch_done, hist,
                                ckpt_cfg, initial_params_vector)
    else:
        epoch_callback = None
    start = time.time()
    opt_params, history = train_model((opt, loss, n_epochs, batch_size), model, params,
                                      train_inputs, train_outputs,
                                      val_inputs, val_outputs,
                                      shuffle_key=shuffle_key,
                                      opt_state=opt_state,
                                      loss_history=history,
                                      start_epoch=start_epoch,
                                      epoch_callback=epoch_callback)
    if ckpt is not None:
        print(f"Training time (resumed from epoch {start_epoch}): {time.time() - start:.2f} seconds")
    else:
        print(f"Training time: {time.time() - start:.2f} seconds")
    if plot_hist:
        plt.figure()
        plt.title(model_name)
        cols = ['firebrick','darkblue']
        i = 0
        for var in ['train_loss', 'val_loss']:
            plt.plot(history[var], label = var,color = cols[i])
            i+=1
        plt.legend()
        plt.show()
    os.makedirs(save_dir, exist_ok=True)
    y_train_hat = model(opt_params, train_inputs)
    y_val_hat = model(opt_params, val_inputs)
    y_train_hat_uz = libs.dataloading.unpreprocess_target(y_train_hat)
    y_val_hat_uz = libs.dataloading.unpreprocess_target(y_val_hat)
    y_train_uz = libs.dataloading.unpreprocess_target(train_outputs)
    y_val_uz = libs.dataloading.unpreprocess_target(val_outputs)
    metrics = {
        'r2_train_scaled': r2_score(train_outputs, y_train_hat),
        'r2_val_scaled': r2_score(val_outputs, y_val_hat),
        'mse_train': mean_squared_error(train_outputs, y_train_hat),
        'mse_val': mean_squared_error(val_outputs, y_val_hat),
        'r2_train': r2_score(y_train_uz, y_train_hat_uz),
        'r2_val': r2_score(y_val_uz, y_val_hat_uz)
    }
    if write_csv:
        with open(f"{save_dir}/QNN_r2.txt", mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([save_postfix, model_name, metrics['r2_train'], metrics['r2_val'], metrics['r2_train_scaled'], metrics['r2_val_scaled'],  metrics['mse_train'], metrics['mse_val']])
    # --- Save ---
    np.savez_compressed(f"{save_dir}/{model_name}_optparams{save_postfix}.npz", opt_params=opt_params)
    np.savez_compressed(f"{save_dir}/{model_name}_history{save_postfix}.npz", history=history)
    if return_optpars:
        return metrics, opt_params, history, initial_params_vector
    else:
        return metrics