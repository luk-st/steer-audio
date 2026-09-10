"""YAML config merging for the steering CLIs.

Precedence is CLI flag > YAML config > argparse default. It is implemented by
feeding the config to ``parser.set_defaults`` and re-parsing ``sys.argv``:
argparse then knows which values the user actually typed, which a
post-parse merge cannot tell (a flag left at its default is indistinguishable
from one passed explicitly with that value).
"""

import argparse
from pathlib import Path
from typing import Sequence


def apply_yaml_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    path_keys: Sequence[str] = (),
) -> argparse.Namespace:
    """Re-parse the command line with ``args.config`` supplying the defaults.

    Args:
        parser: The parser that produced ``args``.
        args: The first-pass parse, used only for ``args.config``.
        path_keys: Dests whose string values become ``Path`` (argparse's
            ``type=`` only runs on strings typed at the CLI, not on defaults).

    Returns:
        A fresh namespace with YAML values applied under any explicit flags.
    """
    import yaml

    with args.config.open() as f:
        cfg = yaml.safe_load(f) or {}
    defaults = {}
    for key, val in cfg.items():
        dest = key.replace("-", "_")
        if not hasattr(args, dest):
            parser.error(f"YAML config has unknown key {key!r}.")
        defaults[dest] = (
            Path(val) if dest in path_keys and isinstance(val, str) else val
        )
    parser.set_defaults(**defaults)
    return parser.parse_args()
