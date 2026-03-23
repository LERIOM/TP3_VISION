import os
import random
from contextlib import nullcontext
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_session_dir(output_root, resume_session):
    resume_path = Path(resume_session)
    if resume_path.is_absolute():
        return resume_path

    candidate_in_output = output_root / resume_path
    if candidate_in_output.exists() or len(resume_path.parts) == 1:
        return candidate_in_output

    return resolve_project_path(resume_session)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_device_backend(device):
    if device.type != "cuda":
        return

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None
