"""Model loading — replicates runner.py's exact load path (config.yaml + model.pt)
locally, since the original runner.py inlines the load and exposes no importable
load_model. The model.pt is a FLAT state_dict, so the ema/model/flat resolution
lands on the flat branch."""
import os
from src.utils.local_paths import bootstrap as _bootstrap, load_paths

_bootstrap()

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pixelflow.utils import config as config_utils  # noqa: E402


def load_model(model_dir=None, device="cuda:0", ckpt_name="model.pt"):
    """Instantiate the PixelFlow model and load weights exactly as runner.py does.
    Returns (model.eval(), omegaconf config). model_dir defaults to paths.model_dir."""
    if model_dir is None:
        model_dir = load_paths()["model_dir"]
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, ckpt_name), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt          # our c2img model.pt -> flat state_dict
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


__all__ = ["load_model"]
