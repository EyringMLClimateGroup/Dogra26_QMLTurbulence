##### Lena Dogra, October 2026
# training QNN models with many different ansätze, submit with sbatch

#!/usr/bin/env python
"""
train_sweep.py -- train PQC regression models for all missing
(key, n_enc, n_dec, local/nonlocal) combinations at a fixed
n_qubits / enc_angles.
Queue-friendly:
  * combos are only skipped if their .npz artifact actually exists on disk,
    so interrupted runs are resumed correctly
  * long models are CONTINUED across re-runs: a checkpoint (params +
    optimizer state + epoch) is written after every epoch and picked up
    where it was interrupted
  * workers do not write the CSV individually; the main process appends rows centrally
    (safe on parallel/network filesystems)
  * one failing model does not kill the job (re-run picks it up)
  * n_jobs defaults to the SLURM CPU allocation
"""
import os, sys
import re
import csv
import time
import argparse
import traceback
# --------------------------------------------------------------------
# env setup: MUST happen before jax/numpy are imported
# --------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")      # no BLAS oversubscription
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import matplotlib
matplotlib.use("Agg")   # headless: workers must never try to show plots
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pennylane as qml
from joblib import Parallel, delayed
sys.path.append('../')
from libs import circuits, dataloading, qnn_models
# share JAX's on-disk compilation cache between workers
# (JAX >= 0.4.19 + shared filesystem; silently skipped otherwise)
try:
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join("/scratch/b/b309252", "jit_cache"))
except Exception:
    pass
# ----------------------------- config ---------------------------------
SAVE_DIR      = "/home/b/b309252/turbulence-parameterization-palm/results/QNN_results"
DEDUP_RESULTS = os.path.join(SAVE_DIR, "QNN_r2.txt")  # done combos
RAW_RESULTS   = os.path.join(SAVE_DIR, "QNN_r2.txt")  # done combos
CKPT_DIR      = os.path.join(SAVE_DIR, "checkpoints") # NEW: per-combo checkpoints
CKPT_EVERY    = 1          # NEW: save a checkpoint after every N epochs
CASE_ID       = 0            # first argument to dataloading.load_data
VAR_ANGLES    = "XYZ"
# shared by all loky workers (pickled once per worker)
DATA = {}
# ----------------------------- helpers --------------------------------
def clear_jax_caches():
    """Clear compiled programs between models.
    jax.clear_jit_cache() needs a newer JAX; on older versions this falls
    back to a gc pass, which still lets the XLA buffer pool reclaim and
    reuse the freed memory once the jitted functions go out of scope.
    """
    try:
        jax.clear_jit_cache()
    except AttributeError:
        import gc
        gc.collect()
def key_to_str(key):
    return "key_" + "_".join(str(int(x)) for x in np.asarray(key).ravel())
def parse_key(s):
    """Restore a PRNGKey from its saved string.
    Handles the clean format ("key_1234_5678") and the legacy str(jax_key)
    format ("Array([1234, 5678], dtype=uint32)"). For the legacy format only
    the bracketed values are read -- the old regex would also grab the '32'
    from 'uint32' and produce a 3-element key.
    """
    s = str(s)
    m = re.search(r"key_([0-9_]+)", s)
    if not m:
        m = re.search(r"\[([0-9_,\s]+)\]", s)
    if not m:
        m = re.search(r"([0-9_,\s]+)", s)
    return jnp.array([int(x) for x in re.findall(r"\d+", m.group(1))], dtype=jnp.uint32)
def parse_key_tuple(s):
    """key string -> hashable tuple of ints (None if unparseable)."""
    try:
        return tuple(int(x) for x in parse_key(s).tolist())
    except Exception:
        return None
def parse_config(s):
    """model_name -> (n_qubits, n_enc, n_dec, entanglement, var_angles, enc_angles).
    Column order must match the assignment in main().
    """
    m = re.match(r"(\d+)qubits_circuit_([a-zA-Z]+)_([a-zA-Z]+)_freq(\d+)enc_(\d+)dec_([a-zA-Z]+)", s)
    if not m:
        return (None,) * 6
    n_q, enc_a, var_a, n_e, n_d, ent = m.groups()
    return int(n_q), int(n_e), int(n_d), ent, var_a, enc_a
def artifact_exists(model_name, key_str):
    """True if the trained-params file for this combo is actually on disk."""
    return os.path.isfile(os.path.join(SAVE_DIR,
                                       f"{model_name}_optparams{key_str}.npz"))
def ckpt_path(model_name, key_str):          # NEW
    return os.path.join(CKPT_DIR, f"{model_name}_ckpt_{key_str}.pkl")
def remove_ckpt(model_name, key_str):        # NEW
    p = ckpt_path(model_name, key_str)
    if os.path.isfile(p):
        os.remove(p)
# ----------------------------- training -------------------------------
def train_one(key, n_qubits, n_enc, n_dec, local_observable, n_epochs, lr):
    circuit = circuits.PQCircuit(n_enc, n_dec, n_qubits,
                                 var_angles=VAR_ANGLES,
                                 encoding_angles=_ENC_ANGLES,
                                 choose_freq=True)
    dev = qml.device("default.qubit", wires=n_qubits)
    model, no_params, model_name = qnn_models.make_model(dev, circuit, local_observable)
    key_str = key_to_str(key)
    metrics = qnn_models.train_and_evaluate(
        model, model_name, no_params, key,
        DATA["train_x"], DATA["train_y"], DATA["val_x"], DATA["val_y"],
        SAVE_DIR,
        save_postfix=key_str,
        plot_hist=False,
        n_epochs=n_epochs,
        learning_rate=lr,
        write_csv=False,   # CSV rows are written centrally by the main process
        # NEW: checkpoint + resume
        ckpt_path=ckpt_path(model_name, key_str),
        save_ckpt_every=CKPT_EVERY,
        ckpt_tag=f"case{CASE_ID}_all_{_ENC_ANGLES}_{DATA['train_x'].shape}",
    )
    return metrics, model_name
_EPOCHS = None
_LR = None
_ENC_ANGLES = None
def run_job(job):
    key, n_qubits, n_enc, n_dec, local_observable = job
    key_str = key_to_str(key)
    t0 = time.time()
    try:
        metrics, model_name = train_one(key, n_qubits, n_enc, n_dec,
                                        local_observable, _EPOCHS, _LR)
    except Exception as e:
        print(f"[pid {os.getpid()}] FAILED n_enc={n_enc} n_dec={n_dec} "
              f"local={local_observable} key={key_str}: {e}", flush=True)
        traceback.print_exc()
        clear_jax_caches()
        return None
    # central CSV append: only the main process writes QNN_r2.txt
    with open(RAW_RESULTS, mode="a", newline="") as f:
        csv.writer(f).writerow([
            key_str, model_name,
            metrics['r2_train'], metrics['r2_val'],
            metrics['r2_train_scaled'], metrics['r2_val_scaled'],
            metrics['mse_train'], metrics['mse_val'],
        ])
    remove_ckpt(model_name, key_str)    # NEW: finished -> artifact is the source of truth
    print(f"[pid {os.getpid()}] done ({time.time()-t0:6.0f}s) n_enc={n_enc} "
          f"n_dec={n_dec} local={local_observable} r2_val={metrics['r2_val']:.4f}",
          flush=True)
    clear_jax_caches()
    return metrics
# ----------------------------- main -----------------------------------
def main():
    global _EPOCHS, _LR, _ENC_ANGLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qubits", type=int, default=2)
    ap.add_argument("--enc-angles", type=str, default="XYZ",
                    choices=["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"],
                    help="encoding angle set (e.g. X, XY, XYZ)")
    ap.add_argument("--n-jobs",   type=int, default=None,
                    help="parallel workers (default: SLURM allocation)")
    ap.add_argument("--epochs",   type=int, default=150)
    ap.add_argument("--lr",       type=float, default=0.0005)
    args = ap.parse_args()
    args.enc_angles = args.enc_angles.upper()
    _EPOCHS, _LR, _ENC_ANGLES = args.epochs, args.lr, args.enc_angles
    n_jobs = args.n_jobs or int(
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_NPROCS")
        or min(os.cpu_count() or 4, 8)
    )
    t_start = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)    # NEW
    DATA["train_x"],  DATA["val_x"], DATA["train_y"],  DATA["val_y"] = \
        dataloading.load_data(CASE_ID, n_samples="all")
    # keys to use this run
    KEY_VALUES = [[1763606390, 1008848841],
                  [ 392068016, 1989062420],
                  [1220103521, 3186102496],
                  [ 995714853, 1645473629],
                  [ 309040335, 1318072158],
                  [1726967592, 4063262276],
                  [2275778133, 2722316996],
                  [3962408671, 2610656276],
                  [   84735912,  977563568],
                  [ 822871285,  221743824]]
    keys = [jnp.array(k, dtype=jnp.uint32) for k in KEY_VALUES]
    key_tuples  = [tuple(k.tolist()) for k in keys]
    key_tuple_set = set(key_tuples)
    # which combos are already done (AND actually saved to disk).
    # Identity is the key VALUE itself: (key_tuple, n_enc, n_dec, local)
    existing = set()
    dfq = pd.read_csv(DEDUP_RESULTS, header=0, index_col=None)
    if len(dfq) > 0:
        dfq[["n_qubits", "n_enc", "n_dec", "entanglement", "angles", "enc_angles"]] = \
            dfq["model_name"].apply(lambda s: pd.Series(parse_config(s)))
        dfq["local_flag"] = (dfq.entanglement == "local").astype(int)
        dfq["key_t"] = dfq["key"].astype(str).map(parse_key_tuple)
        # rows from this run's key set, same n_qubits and enc_angles
        df1 = dfq[(dfq.n_qubits == args.n_qubits)
                  & (dfq.enc_angles.str.upper() == args.enc_angles)
                  & dfq.key_t.map(lambda t: t in key_tuple_set)].copy()
        for r in df1.itertuples(index=False):
            if r.n_enc is None or r.n_enc != r.n_enc:      # unparseable row -> skip
                continue
            if artifact_exists(r.model_name, str(r.key)):
                existing.add((r.key_t, int(r.n_enc), int(r.n_dec),
                              int(r.local_flag)))
                remove_ckpt(r.model_name, str(r.key))      # NEW: prune stale ckpt
            else:
                print(f"WARNING: CSV row but missing artifact, will retrain: "
                      f"{r.model_name} (key {r.key})", flush=True)
    # build the job list
    jobs = []
    for n_dec in range(0, 5):
        for n_enc in range(1, 10):
            for local_observable in (0, 1):
                for key, key_t in zip(keys, key_tuples):
                    if (key_t, n_enc, n_dec, local_observable) in existing:
                        continue
                    jobs.append((key, args.n_qubits, n_enc, n_dec, local_observable))
    n_total = len(keys) * 9 * 5 * 2
    n_ckpts = len([f for f in os.listdir(CKPT_DIR) if f.endswith(".pkl")])  # NEW
    print(f"n_qubits={args.n_qubits}, enc_angles={args.enc_angles}: "
          f"{len(jobs)} jobs to run ({len(existing)} done, {n_total} total), "
          f"n_jobs={n_jobs}, {n_ckpts} checkpoints on disk", flush=True)
    if not jobs:
        return
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1, verbose=1)(
        delayed(run_job)(job) for job in jobs)
    ok = [r for r in results if r is not None]
    print(f"\nfinished: {len(ok)} ok, {len(results) - len(ok)} failed, "
          f"{time.time() - t_start:.0f}s total", flush=True)
    if ok:
        print(pd.DataFrame(ok)[["r2_train", "r2_val"]].describe())

if __name__ == "__main__":
    main()