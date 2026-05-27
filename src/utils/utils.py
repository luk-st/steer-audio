import json
import os
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.io.wavfile import write

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg, *args, **kwargs)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def save_audio_results(audio: np.ndarray | torch.Tensor, output_path: str) -> None:
    """
    Saves audio data to .wav files in the specified output path.

    Args:
        audio (np.ndarray | torch.Tensor): Audio data to save.
        output_path (str): Path to the directory where results will be saved.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy().squeeze()

    results_path = os.path.join(output_path, "results")
    os.makedirs(results_path, exist_ok=True)

    for idx, clip in enumerate(audio):
        output_file = os.path.join(results_path, f"audio_{idx}.wav")
        # Assuming audio is normalized between -1.0 and 1.0, rescale to int16
        scaled_audio = (clip * 32767).astype("int16")
        write(output_file, rate=16000, data=scaled_audio)


def save_dict_to_json(dictionary: Dict, file_path: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Args:
        dictionary (Dict): The dictionary to save.
        file_path (str): The path where the JSON file will be saved.
    """
    with open(file_path, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)
