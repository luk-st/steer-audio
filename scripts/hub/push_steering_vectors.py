"""Upload one CAA steering-vector directory to its own HuggingFace Hub repo.

One repo per checkpoint — this preserves per-checkpoint download stats and lets
each artifact be linked individually from the paper page.

Usage::

    python scripts/hub/push_steering_vectors.py \
        --src steering_vectors/caa/ace_piano \
        --repo-id <hf-org>/ace-step-caa-piano \
        --tags ace-step caa piano
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.steering.hub import SteeringVectorArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Local SV directory")
    parser.add_argument("--repo-id", required=True, help="<org>/<repo-name>")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=(),
        help="Extra discovery tags appended to the model card.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload CAA steering vectors",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source directory not found: {args.src}")

    artifact = SteeringVectorArtifact.from_dir(args.src)
    url = artifact.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        tags=args.tags,
    )
    print(f"Uploaded -> {url}")


if __name__ == "__main__":
    main()
