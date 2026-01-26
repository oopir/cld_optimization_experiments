import subprocess
import torch

def _get_idle_gpus(util_threshold=5):
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], encoding="utf-8")
    except Exception:
        return None

    utils = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            utils.append(int(line))
        except ValueError:
            return None

    idle = [i for i, u in enumerate(utils) if u <= util_threshold]
    return idle[:5]

def select_idle_gpus_for_experiment(device="cuda", util_threshold=5):
    """
    Decide once which GPUs to use for the whole experiment.
    - For CPU: returns [None].
    - For explicit device "cuda:k": returns [k] (no utilization check).
    - For "cuda": returns list of idle GPU indices, or raises if none idle.
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return [None]

    if ":" in device:
        idx = int(device.split(":", 1)[1])
        return [idx]

    idle = _get_idle_gpus(util_threshold=util_threshold)
    if idle is None:
        raise RuntimeError("Could not query GPU utilization via nvidia-smi.")
    if not idle:
        raise RuntimeError("No idle GPUs found under utilization threshold.")
    return idle
