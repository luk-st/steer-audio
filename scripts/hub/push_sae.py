"""Upload one SAE checkpoint to its own HuggingFace Hub repo.

One repo per checkpoint so each model gets its own download counter and paper-page
link. For multi-hookpoint releases pass ``--hookpoint`` to upload into a subfolder
(matching what ``Sae.load_from_hub(hookpoint=...)`` expects).

Usage::

    python scripts/hub/push_sae.py \
        --ckpt checkpoints/sae/transformer_blocks.7.cross_attn \
        --repo-id <hf-org>/ace-step-sae-tf7-cross-attn \
        --hookpoint transformer_blocks.7.cross_attn \
        --tags ace-step sae sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` and ``steering`` importable from repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.steering.methods.sae.lib.sae.sae import Sae
from src.steering.hub import push_sae_to_hub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, required=True, help="Local SAE checkpoint dir"
    )
    parser.add_argument("--repo-id", required=True, help="<org>/<repo-name>")
    parser.add_argument(
        "--hookpoint",
        default=None,
        help="Optional sub-directory name inside the repo (e.g. 'transformer_blocks.7.cross_attn').",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--tags", nargs="*", default=())
    parser.add_argument("--commit-message", default="Upload SAE checkpoint")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint directory not found: {args.ckpt}")

    sae = Sae.load_from_disk(args.ckpt)
    url = push_sae_to_hub(
        sae,
        repo_id=args.repo_id,
        hookpoint=args.hookpoint,
        private=args.private,
        tags=args.tags,
        commit_message=args.commit_message,
    )
    print(f"Uploaded -> {url}")


if __name__ == "__main__":
    main()
