import argparse
import shutil
from pathlib import Path


def del_audios_dir(path: Path | str) -> None:
    """Deletes the audios directory."""
    if isinstance(path, str):
        path = Path(path)
    assert path.is_dir()
    print(f"Deleting audios directory: {path}")
    shutil.rmtree(path)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    del_audios_dir(args.path)
