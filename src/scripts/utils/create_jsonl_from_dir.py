import argparse
from pathlib import Path


def main(dirpath: str, filename: str):
    assert filename.endswith(".jsonl")
    assert Path(dirpath).is_dir()

    dirpath = Path(dirpath)
    outfile = (dirpath.parent.parent / filename).resolve()

    all_files = list(dirpath.glob("*.wav"))
    all_files = sorted(all_files)

    with open(outfile, "w") as f:
        for file in all_files:
            f.write('{"path":"' + str(file) + '"}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--filename", type=str, default="audios.jsonl")
    args = parser.parse_args()
    main(args.dir, args.filename)