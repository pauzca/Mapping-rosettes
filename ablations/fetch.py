"""Download the reproduction data (test split, checkpoints, orthomosaics) from the GitHub release.

The release holds one tarball per Kaggle dataset (mixed-resolution-frailejones,
low-resolution-frailejones-detection) with dataset/data.yaml, dataset/test/, checkpoints/ and, for the
mixed one, mosaics/. Each tarball is checked against SHA256SUMS and extracted to <out>/<dataset>/.
While the repository is private, set GITHUB_TOKEN to a token with read access to it.

Example:
    GITHUB_TOKEN=... python -m ablations.fetch --out data
"""
import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path

REPO, TAG = "pauzca/Mapping-rosettes", "ablation-v1"


def request(url, token, accept="application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "mapping-rosettes"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def download(asset, dest, token):
    with request(asset["url"], token, accept="application/octet-stream") as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if all((out / name / "dataset" / "data.yaml").exists()
           for name in ("mixed-resolution-frailejones", "low-resolution-frailejones-detection")):
        print(f"data already present in {out}")
        return

    release = json.load(request(f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}", token))
    assets = {a["name"]: a for a in release["assets"]}
    download(assets["SHA256SUMS"], out / "SHA256SUMS", token)
    sums = dict(line.split()[::-1] for line in (out / "SHA256SUMS").read_text().splitlines() if line.strip())
    for name, digest in sums.items():
        target = out / name.removesuffix(".tar")
        if (target / "dataset" / "data.yaml").exists():
            print(f"{target} already present")
            continue
        tar = out / name
        print(f"downloading {name} ({assets[name]['size'] / 1e6:.0f} MB)", flush=True)
        download(assets[name], tar, token)
        if sha256(tar) != digest:
            raise SystemExit(f"checksum mismatch for {name}")
        with tarfile.open(tar) as t:
            t.extractall(target, filter="data")
        tar.unlink()
        print(f"extracted to {target}")


if __name__ == "__main__":
    main()
