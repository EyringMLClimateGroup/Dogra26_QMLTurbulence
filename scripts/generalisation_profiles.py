### Lena Dogra, October 2026
#!/usr/bin/env python
"""
generalisation_profiles.py -- recalculate vertical profiles of the QNN prediction
error for all trained QNN models and data series.

Queue-friendly:
  * progress is pickled after every chunk, so interrupted runs resume
  * 'missing_params' tasks (optparams file absent) are retried on re-run
  * one failing model does not kill the job
"""
import os, sys
import time
import pickle
import argparse

# --------------------------------------------------------------------
# env setup: MUST happen before jax/numpy are imported
# --------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import matplotlib
matplotlib.use("Agg")   # headless: workers must never try to show plots

import numpy as np
import pandas as pd
import jax.numpy as jnp
import pennylane as qml
from joblib import Parallel, delayed
sys.path.append('../')
from libs import circuits, dataloading, qnn_models

# share JAX's on-disk compilation cache between workers
try:
    import jax
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join("/scratch/b/b309252", "jit_cache"))
except Exception:
    pass

# ----------------------------- config ---------------------------------
RESULTS_DIR   = "../results/QNN_results/"
RESULTS_CSV   = os.path.join(RESULTS_DIR, "QNN_r2.txt")
OUT_FILE      = os.path.join(RESULTS_DIR, "panel_b_data_qu.pkl")
MISSING_FILE  = os.path.join(RESULTS_DIR, "panel_b_data_qu_missing.pkl")
ERROR_FILE    = os.path.join(RESULTS_DIR, "panel_b_data_qu_errors.pkl")
DONE_FILE     = os.path.join(RESULTS_DIR, "panel_b_data_qu_done_tasks.pkl")

BIN_EDGES     = np.linspace(0.2, 0.8, 20)
SERIES_LIST   = [f"d{i}" for i in range(1, 7)]
VAR_ANGLES    = "XYZ"
N_SAMPLES     = 30000
CONFIGS       = [[2, "XYZ"], [3, "XY"], [4, "XYZ"],
                 [6, "X"], [6, "XY"], [8, "XY"]]

# ----------------------------- helpers --------------------------------
def atomic_pickle_dump(obj, filename):
    tmp = filename + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, filename)

def load_pickle_safe(fname):
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        with open(fname, "rb") as f:
            return pickle.load(f)
    return []

def deduplicate(records):
    seen = {}
    for r in records:
        seen[r["task_id"]] = r
    return list(seen.values())

def bin_by_height(zs, y, bin_edges):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    out = np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        m = (zs >= bin_edges[i]) & (zs < bin_edges[i + 1])
        if m.sum() > 1:
            out[i] = np.mean(y[m])
    return bin_centers, out

def make_model_name(n_enc, n_dec, n_qubits, local_observable, encoding_angles):
    circuit = circuits.PQCircuit(n_enc, n_dec, n_qubits,
                                 var_angles=VAR_ANGLES,
                                 encoding_angles=encoding_angles,
                                 choose_freq=True)
    return (str(circuit.n_wires) + "qubits_" + circuit.circuit_name
            + str(n_enc) + "enc_" + str(n_dec) + "dec_"
            + ("" if local_observable else "non") + "local")

def load_model(n_qubits, n_enc, n_dec, local_observable, encoding_angles,
               key_str):
    model_qnn, _, model_name = qnn_models.make_model(
        qml.device("default.qubit", wires=n_qubits),
        circuits.PQCircuit(n_enc, n_dec, n_qubits,
                           var_angles=VAR_ANGLES,
                           encoding_angles=encoding_angles,
                           choose_freq=True),
        local_observable,
    )
    loaded = np.load(f"{RESULTS_DIR}{model_name}_optparams{key_str}.npz")
    return model_qnn, loaded["opt_params"], model_name

# ----------------------------- worker ---------------------------------
def run_task(task_id, ser, key, local_observable, n_qubits, enc_angles,
             n_enc, n_dec):
    
    opt_file = (f"{RESULTS_DIR}"
                f"{make_model_name(n_enc, n_dec, n_qubits, local_observable, enc_angles)}"
                f"_optparams{key}.npz")

    base = dict(task_id=task_id, series=ser, key=key,
                local_observable=local_observable, n_qubits=n_qubits,
                enc_angles=enc_angles, n_enc=n_enc, n_dec=n_dec)
    if not os.path.exists(opt_file):
        return dict(**base, status="missing_params")
    try:
        jnp_x, jnp_y = dataloading.load_gen_data(
            0, series_to_load=[ser], n_samples=N_SAMPLES)
        zs = np.asarray(jnp_x[:, 2]) / np.pi + 0.5
        model_qnn, opt_params, model_name = load_model(
            n_qubits, n_enc, n_dec, local_observable, enc_angles, key)
        y_hat = model_qnn(opt_params, jnp_x)
        zc, mse_qu = bin_by_height(zs, (y_hat - jnp_y) ** 2, BIN_EDGES)
        return dict(**base, status="ok", model=model_name,
                    z=zc, mse_quantum=mse_qu)
    except FileNotFoundError as e:
        print(f"[pid {os.getpid()}] missing optparams: {e}", flush=True)
        return dict(**base, status="missing_params")
    except Exception as e:
        print(f"[pid {os.getpid()}] FAILED ser={ser} key={key} "
              f"nq={n_qubits} enc={n_enc} dec={n_dec} "
              f"local={local_observable}: {e}", flush=True)
        return dict(**base, status="error", error=repr(e))

# ----------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs",    type=int, default=None,
                    help="parallel workers (default: SLURM allocation)")
    ap.add_argument("--chunk-size", type=int, default=100,
                    help="tasks per save step (default: 100)")
    args = ap.parse_args()

    n_jobs = args.n_jobs or int(
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_NPROCS")
        or min(os.cpu_count() or 4, 8)
    )
    t_start = time.time()

    # keys: the raw CSV strings -- exactly the strings used in the
    # optparams filenames, so they serve directly as save_postfix
    dfq = pd.read_csv(RESULTS_CSV, header=0, index_col=None)
    keys = list(dfq["key"].astype(str).unique())
    print(f"found {len(keys)} unique keys", flush=True)

    tasks = [
        (i, ser, key, local_observable, n_qubits, enc_angles, n_enc, n_dec)
        for i, (ser, key, local_observable, n_qubits, enc_angles,
                n_enc, n_dec) in enumerate(
            (ser, key, local_observable, n_qubits, enc_angles, n_enc, n_dec)
            for ser in SERIES_LIST
            for key in keys
            for local_observable in [0, 1]
            for (n_qubits, enc_angles) in CONFIGS
            for n_enc in range(1, 10)
            for n_dec in range(5)
        )
    ]
    print("Total tasks:", len(tasks), flush=True)

    panel_b_data = deduplicate(load_pickle_safe(OUT_FILE))
    missing_data = deduplicate(load_pickle_safe(MISSING_FILE))
    error_data   = deduplicate(load_pickle_safe(ERROR_FILE))
    print(f"Loaded: ok={len(panel_b_data)} "
          f"missing={len(missing_data)} errors={len(error_data)}", flush=True)

    done_task_ids = set(r["task_id"] for r in panel_b_data)
    remaining_tasks = [t for t in tasks if t[0] not in done_task_ids]
    print(f"Remaining tasks: {len(remaining_tasks)}, n_jobs={n_jobs}",
          flush=True)
    if not remaining_tasks:
        print("Nothing to do.", flush=True)
        return

    chunk_size = args.chunk_size
    for start in range(0, len(remaining_tasks), chunk_size):
        chunk = remaining_tasks[start:start + chunk_size]
        chunk_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            delayed(run_task)(*task) for task in chunk)
        for r in chunk_results:
            task_id = r["task_id"]
            if r["status"] == "ok":
                done_task_ids.add(task_id)
                panel_b_data.append(r)
                missing_data = [m for m in missing_data
                                if m["task_id"] != task_id]
                error_data   = [e for e in error_data
                                if e["task_id"] != task_id]
            elif r["status"] == "missing_params":
                missing_data = [m for m in missing_data
                                if m["task_id"] != task_id]
                missing_data.append(r)
            else:
                error_data = [e for e in error_data
                              if e["task_id"] != task_id]
                error_data.append(r)

        panel_b_data = deduplicate(panel_b_data)
        missing_data = deduplicate(missing_data)
        error_data   = deduplicate(error_data)
        atomic_pickle_dump(panel_b_data, OUT_FILE)
        atomic_pickle_dump(missing_data, MISSING_FILE)
        atomic_pickle_dump(error_data, ERROR_FILE)
        atomic_pickle_dump(sorted(done_task_ids), DONE_FILE)
        print(f"Saved chunk {start // chunk_size + 1} | "
              f"ok={len(panel_b_data)} missing={len(missing_data)} "
              f"errors={len(error_data)} "
              f"({time.time() - t_start:.0f}s)", flush=True)

    print("Done.", flush=True)

if __name__ == "__main__":
    main()