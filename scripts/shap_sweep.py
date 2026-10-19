### Lena Dogra, October 2026
## calculation of SHAP values for all QNN, using ExactExplainer, submit with sbatch

#!/usr/bin/env python
"""
shap_sweep.py -- compute SHAP values for every QNN model that has optparams
on disk. Queue-friendly:
  * only models with an optparams file are processed; already-done combos
    (shap .npz present) are skipped, so it resumes after preemption
  * workers do not write anything except their own .npz
  * n_jobs defaults to the SLURM CPU allocation
"""
import os, sys
import re
import time
import argparse
import traceback

# --------------------------------------------------------------------
# env setup: MUST happen before jax/numpy/tf are imported
# --------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib
matplotlib.use("Agg")   # headless: never try to show plots

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pennylane as qml
import shap
from functools import reduce
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.append('../')
from libs import circuits, dataloading, qnn_models

# share the JAX on-disk compile cache between workers (same circuit shapes
# across different keys reuse one compile)
try:
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join("/scratch/b/b309252", "jit_cache"))
except Exception:
    pass

# ----------------------------- config ---------------------------------
SAVE_DIR    = "/home/b/b309252/turbulence-parameterization-palm/results/QNN_results/"
RESULTS_CSV = os.path.join(SAVE_DIR, "QNN_r2.txt")
CASE_ID     = 0
VAR_ANGLES  = "XYZ"
FEATURE_NAMES = ['$p$', '$\\theta$', '$z$/ABL', '$u$', '$v$', '$w$']

# (n_qubits, enc_angles) configs that exist in the runs
CONFIGS = [(2, "XYZ"), (3, "XY"), (4, "XYZ"), (6, "X"), (6, "XY"), (8, "XY")]

# shared by all loky workers (pickled once per worker)
DATA = {}

# ----------------------------- helpers --------------------------------
def key_to_str(key):
    return "key_" + "_".join(str(int(x)) for x in np.asarray(key).ravel())

def key_variants(key):
    """Both filename spellings the optparams file may have been saved under."""
    a = [int(x) for x in np.asarray(key).ravel()]
    clean  = key_to_str(key)
    legacy = "Array([" + ", ".join(str(x) for x in a) + "], dtype=uint32)"
    return [clean, legacy]

def parse_key(s):
    """Restore a PRNGKey from its saved string (clean or legacy format)."""
    s = str(s)
    m = re.search(r"key_([0-9_]+)", s)
    if not m:
        m = re.search(r"\[([0-9_,\s]+)\]", s)
    if not m:
        m = re.search(r"([0-9_,\s]+)", s)
    return jnp.array([int(x) for x in re.findall(r"\d+", m.group(1))], dtype=jnp.uint32)

def make_model_name(n_enc, n_dec, n_qubits, local_observable, enc_angles):
    """Must stay identical to qnn_models.make_model's name formula."""
    circuit = circuits.PQCircuit(n_enc, n_dec, n_qubits,
                                 var_angles=VAR_ANGLES,
                                 encoding_angles=enc_angles,
                                 choose_freq=True)
    return (str(circuit.n_wires) + "qubits_" + circuit.circuit_name
            + str(circuit.n_enc) + "enc_" + str(circuit.n_dec) + "dec_"
            + ("" if local_observable else "non") + "local")

def find_optparams(model_name, key):
    """Return the optparams path if it exists (clean or legacy key name)."""
    for cand in key_variants(key):
        p = os.path.join(SAVE_DIR, f"{model_name}_optparams{cand}.npz")
        if os.path.isfile(p):
            return p
    return None

def extract_shap(explainer, sample):
    """Version-robust SHAP extraction.

    New unified SHAP: explainer(X) -> Explanation(.values / .base_values).
      - batched call first, per-row fallback if the batch path complains
    Legacy SHAP:      explainer.shap_values(X), .expected_value
    Returns (values, base) with values of shape (n_samples, n_features).
    """
    if hasattr(explainer, "shap_values"):
        values = np.asarray(explainer.shap_values(sample))
        base = getattr(explainer, "expected_value", None)
    else:
        try:
            expl = explainer(sample)
            values = np.asarray(getattr(expl, "values", expl))
            base = getattr(expl, "base_values", None)
            if base is None:
                base = getattr(expl, "expected_value", None)
        except Exception as e:
            print(f"  batched explain failed ({type(e).__name__}: {e}); "
                  f"falling back to per-row", flush=True)
            vals, base = [], None
            for row in sample:
                expl = explainer(np.asarray(row, dtype=float)[None, :])
                vals.append(np.asarray(getattr(expl, "values", expl)).reshape(-1))
                if base is None:
                    b = getattr(expl, "base_values", None)
                    if b is None:
                        b = getattr(expl, "expected_value", None)
                    if b is not None:
                        base = float(np.asarray(b).reshape(-1)[0])
            values = np.stack(vals)
    # unified API may give (n, f) or (n, f, 1) for a single output
    if values.ndim == 3:
        values = values[:, :, 0]
    if base is not None:
        base = float(np.asarray(base).reshape(-1)[0])
    return values, base

# ----------------------------- SHAP -----------------------------------
def shap_analysis(n_enc, n_dec, n_qubits, local_observable, key, enc_angles,
                  opt_file, bg_size, sample_size):
    key_str = key_to_str(key)
    circuit = circuits.PQCircuit(n_enc, n_dec, n_qubits,
                                 var_angles=VAR_ANGLES,
                                 encoding_angles=enc_angles,
                                 choose_freq=True)
    model_name = make_model_name(n_enc, n_dec, n_qubits, local_observable, enc_angles)
    dev = qml.device("default.qubit", wires=n_qubits)

    pars = np.load(opt_file)["opt_params"]

    if local_observable:
        obs_return = lambda: [qml.expval(qml.PauliZ(i)) for i in range(circuit.n_wires)]
    else:
        full_obs = reduce(lambda x, y: x @ y,
                          [qml.PauliZ(i) for i in range(circuit.n_wires)])
        obs_return = lambda: [qml.expval(full_obs)]

    @qml.qnode(dev, interface="jax")
    def qnn_pqc(inputs, angles):
        circuit(inputs, angles, dev.wires)
        return obs_return()

    def qnn_single(x, angles):
        x = x[None, :]
        return qnn_pqc(x, angles)[0]     # single (first) output

    qnn_batch = jax.jit(jax.vmap(lambda x: qnn_single(x, pars)))

    def qnn_model(X):
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        # IMPORTANT: explicit 2-D output (n, 1) -- the unified SHAP API
        # infers batch vs per-row dims from the output shape, and a bare
        # (n,) makes it iterate over scalars.
        return np.asarray(qnn_batch(X)).reshape(X.shape[0], 1)

    background = np.asarray(DATA["background"])[:bg_size]
    sample = np.asarray(DATA["sample"])[:sample_size]

    # loud guard: both must be 2-D (n, n_features) with matching features
    if background.ndim != 2 or sample.ndim != 2 or background.shape[1] != sample.shape[1]:
        raise ValueError(
            f"bad data shapes: background={background.shape}, sample={sample.shape}")

    print(f"  [shap] {model_name}: bg={background.shape} "
          f"sample={sample.shape}", flush=True)

    explainer = shap.ExactExplainer(qnn_model, background)
    shap_values, base_value = extract_shap(explainer, sample)
    shap_values = np.asarray(shap_values)

    out_npz = f"{SAVE_DIR}{model_name}_shap{key_str}_exact.npz"
    np.savez(out_npz, shap_values=shap_values,
             base_value=(base_value if base_value is not None else -1.0))

    return out_npz

def run_one(job):
    (n_enc, n_dec, n_qubits, local_observable, enc_angles,
     key, opt_file, bg_size, sample_size) = job
    key_str = key_to_str(key)
    t0 = time.time()
    try:
        out = shap_analysis(n_enc, n_dec, n_qubits, local_observable, key,
                            enc_angles, opt_file, bg_size, sample_size)
    except Exception as e:
        print(f"[pid {os.getpid()}] FAILED n_enc={n_enc} n_dec={n_dec} "
              f"nq={n_qubits} local={local_observable} key={key_str}: {e}",
              flush=True)
        traceback.print_exc()
        return None
    print(f"[pid {os.getpid()}] done ({time.time()-t0:6.0f}s) n_enc={n_enc} "
          f"n_dec={n_dec} nq={n_qubits} local={local_observable} "
          f"key={key_str} -> {out}", flush=True)
    return out

# ----------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits",   type=int,   default=None,
                    help="restrict to this n_qubits (default: all configs)")
    ap.add_argument("--enc-angles", type=str,   default=None,
                    choices=[None, "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"],
                    help="restrict to this enc_angles (default: all configs)")
    ap.add_argument("--bg-size",    type=int,   default=200)
    ap.add_argument("--sample-size",type=int,   default=1000)
    ap.add_argument("--n-jobs",     type=int,   default=None,
                    help="parallel workers (default: SLURM allocation)")
    args = ap.parse_args()
    if args.enc_angles is not None:
        args.enc_angles = args.enc_angles.upper()

    n_jobs = args.n_jobs or int(
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_NPROCS")
        or min(os.cpu_count() or 4, 8)
    )

    t_start = time.time()
    tr_x, va_x, _tr_y, _va_y = dataloading.load_data(CASE_ID, n_samples="all")
    DATA["background"] = np.asarray(tr_x)
    DATA["sample"]     = np.asarray(va_x)
    print(f"loaded: train_x={DATA['background'].shape} "
          f"val_x={DATA['sample'].shape}", flush=True)

    # unique keys used so far, keyed by value
    key_tuples, key_values = set(), []
    for s in pd.read_csv(RESULTS_CSV)["key"].fillna("").astype(str):
        try:
            t = tuple(int(x) for x in parse_key(s).tolist())
        except Exception:
            continue
        if not t or t in key_tuples:
            continue
        key_tuples.add(t)
        key_values.append(jnp.array(t, dtype=jnp.uint32))
    keys = key_values
    print(f"found {len(keys)} unique keys", flush=True)

    # build the job list: only combos with an optparams file, and no shap file yet
    jobs, n_no_opt, n_done = [], 0, 0
    for n_qubits, enc_angles in CONFIGS:
        if args.n_qubits is not None and n_qubits != args.n_qubits:
            continue
        if args.enc_angles is not None and enc_angles != args.enc_angles:
            continue
        for n_enc in range(1, 10):
            for n_dec in range(0, 10):
                for local_observable in (0, 1):
                    for key in keys:
                        model_name = make_model_name(
                            n_enc, n_dec, n_qubits, local_observable, enc_angles)
                        if os.path.isfile(
                                f"{SAVE_DIR}{model_name}_shap{key_to_str(key)}_exact.npz"):
                            n_done += 1
                            continue
                        opt_file = find_optparams(model_name, key)
                        if opt_file is None:
                            n_no_opt += 1
                            continue
                        jobs.append((n_enc, n_dec, n_qubits, local_observable,
                                     enc_angles, key, opt_file,
                                     args.bg_size, args.sample_size))

    print(f"{len(jobs)} to compute  ({n_done} already done, "
          f"{n_no_opt} no optparams on disk), n_jobs={n_jobs}", flush=True)
    if not jobs:
        return

    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1, verbose=1)(
        delayed(run_one)(job) for job in jobs)

    ok = [r for r in results if r is not None]
    print(f"\nfinished: {len(ok)} ok, {len(results) - len(ok)} failed, "
          f"{time.time() - t_start:.0f}s total", flush=True)

if __name__ == "__main__":
    main()