"""Acquire and verify the three source recording archives."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_destination(name):
    dest = (RAW / name).resolve()
    if RAW.resolve() not in dest.parents and dest != RAW.resolve():
        raise ValueError("Archive entry escapes the destination")


def unzip(path, depth=0):
    if depth > 3:
        raise ValueError("Unexpected nesting depth")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            safe_destination(name)
        archive.extractall(RAW)
    for name in names:
        nested = RAW / name
        if nested.suffix.lower() == ".zip":
            unzip(nested, depth + 1)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "data" / "source_archives.json").read_text())
    for spec in sources:
        path = RAW / spec["filename"]
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".part")
            print("Downloading", spec["filename"], flush=True)
            urllib.request.urlretrieve(spec["url"], temp)
            temp.replace(path)
        if digest(path) != spec["sha256"]:
            raise SystemExit("Source hash differs from the recorded release: " + str(path))
        if path.suffix == ".zip":
            unzip(path)
        else:
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    safe_destination(member.name)
                    if member.issym() or member.islnk():
                        raise ValueError("Unexpected archive link")
                archive.extractall(RAW)
        print("Verified and extracted", spec["filename"], flush=True)


if __name__ == "__main__":
    main()
