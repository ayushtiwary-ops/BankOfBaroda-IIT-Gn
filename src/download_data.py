#!/usr/bin/env python3
"""
PRAMAAN — Reproducible dataset acquisition
==========================================
Downloads every public dataset that backs PRAMAAN's five detections, verifies a
SHA-256 checksum, and lays the data out under ``data/raw/<dataset>/``.

This is the first stage of the reproducibility chain mandated by the project
charter:  ``download_data.py`` -> ``train.py`` -> ``evaluate.py`` regenerate
every reported number from scratch.

Each dataset maps to a specific detection in the problem statement:
    paysim         -> #3 suspicious onboarding / mule + money-flow fraud
    ieee_cis       -> #1 anomalous behaviour + #2 new-device / device trust
    rba            -> #4 suspicious account recovery + login risk (ATO)
    cmu_keystroke  -> #1 behavioural biometrics (keystroke dynamics)
    cert_insider   -> #5 privileged-access misuse (insider threat)

Usage
-----
    python src/download_data.py --all
    python src/download_data.py paysim rba cmu_keystroke
    python src/download_data.py --list

Notes
-----
* PaySim / RBA / CMU are open (no auth). They are fetched over HTTPS.
* IEEE-CIS is a Kaggle *competition* dataset: it needs the Kaggle CLI
  (``pip install kaggle``), credentials at ``~/.kaggle/kaggle.json``, and a
  one-time acceptance of the competition rules on kaggle.com.
* CERT r4.2 is large (multi-GB). We fetch it from the CMU Kilthub record;
  a smaller HF mirror is provided as a fallback.
* Checksums are verified when known. The first time a new file is fetched the
  script prints its SHA-256 so it can be pinned here.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset registry — single source of truth (mirrors DATA_SOURCES.md)
# ---------------------------------------------------------------------------
DATASETS = {
    "paysim": {
        "detection": "#3 mule / money-flow fraud",
        "license": "MIT (mirror) — original PaySim synthetic, Lopez-Rojas et al.",
        "url": "https://huggingface.co/datasets/theman10/paysim/resolve/main/paysim.csv?download=true",
        "out": "paysim/paysim.csv",
        "sha256": "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b",
        "size": 493_534_783,
    },
    "rba": {
        "detection": "#4 account recovery / login risk (ATO)",
        "license": "CC BY 4.0 — Wiefling et al. (das-group)",
        "url": "https://zenodo.org/api/records/6782156/files/rba-dataset.zip/content",
        "out": "rba/rba-dataset.zip",
        "unzip_to": "rba",
        "sha256": "ca1d974e97aebfb30878a613f4ca5c793860a98ba2acb5185c9bc610d7432a33",
        "size": 1_093_700_330,
    },
    "cmu_keystroke": {
        "detection": "#1 behavioural biometrics (keystroke dynamics)",
        "license": "Free for research — Killourhy & Maxion, CMU (DSN 2009)",
        "url": "https://raw.githubusercontent.com/njanakiev/keystroke-biometrics/master/data/DSL-StrongPasswordData.csv",
        "out": "cmu_keystroke/DSL-StrongPasswordData.csv",
        "sha256": "4a7086f601052e307eff24a4bc525c8d104662f8ba06da1ac8080e70a6d55789",
        "size": 4_629_134,
    },
    "ieee_cis": {
        "detection": "#1 anomalous behaviour + #2 new-device / device trust",
        "license": "Competition use — IEEE-CIS / Vesta (Kaggle)",
        "kaggle_competition": "ieee-fraud-detection",
        "out": "ieee_cis/ieee-fraud-detection.zip",
        "unzip_to": "ieee_cis",
        "sha256": "4cc646da09d0a9b265983ffed775b1f9ee15af5266586df610e04d6adae0b829",
        "size": 123_856_947,
    },
    "cert_insider": {
        "detection": "#5 privileged-access misuse (insider threat)",
        "license": "ExactData EUA via CMU CERT (NO redistribution — local use only). Canonical: Kilthub 12841247",
        # Per-file fetch from the HF mirror of CERT r4.2 (jinmang2/cert_insider_threat).
        # http.csv (14.5 GB) and email.csv (1.36 GB) are intentionally excluded from the
        # default sprint set — documented scope decision; add here if needed.
        "files": [
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/r4.2/logon.csv?download=true",
             "cert_insider/r4.2/logon.csv",
             "a770601339829d544535b3cb08b2f2020a1feaa81bd88cdbfc2d452444f12a13", 58_514_706),
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/r4.2/device.csv?download=true",
             "cert_insider/r4.2/device.csv",
             "39925627f0219c77dbb39e5a416715298ef47c2fd25322aa44c66cb128d7291d", 28_982_749),
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/r4.2/file.csv?download=true",
             "cert_insider/r4.2/file.csv",
             "637281bd6d8947b2cccf90bca7096e8cfad711eab1e425c61f886b0e4edfda7f", 193_055_265),
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/r4.2/psychometric.csv?download=true",
             "cert_insider/r4.2/psychometric.csv", None, 43_671),
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/r4.2/readme.txt?download=true",
             "cert_insider/r4.2/readme.txt", None, 6_613),
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/r4.2/license.txt?download=true",
             "cert_insider/r4.2/license.txt", None, 3_890),
            ("https://huggingface.co/datasets/jinmang2/cert_insider_threat/resolve/main/answers/insiders.csv?download=true",
             "cert_insider/answers/insiders.csv", None, 13_792),
        ],
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = done / total * 100
        sys.stdout.write(f"\r    {done/1e6:8.1f} MB / {total/1e6:.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r    {done/1e6:8.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pramaan-downloader/1.0"})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _progress(done, total)
    sys.stdout.write("\n")
    tmp.replace(dest)


def verify(path: Path, expected: str | None) -> None:
    digest = sha256(path)
    if expected is None:
        print(f"    sha256 (pin this in DATASETS): {digest}")
    elif digest != expected:
        raise SystemExit(f"!! checksum mismatch for {path.name}\n   got      {digest}\n   expected {expected}")
    else:
        print(f"    checksum OK ({digest[:16]}…)")


def unzip(path: Path, to: Path) -> None:
    to.mkdir(parents=True, exist_ok=True)
    print(f"    unzip -> {to}")
    with zipfile.ZipFile(path) as z:
        z.extractall(to)


def fetch_kaggle(spec: dict, out: Path) -> None:
    comp = spec["kaggle_competition"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"    kaggle competitions download -c {comp}")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(out.parent)],
            check=True,
        )
    except FileNotFoundError:
        raise SystemExit("!! kaggle CLI not found. `pip install kaggle` and place ~/.kaggle/kaggle.json")
    except subprocess.CalledProcessError:
        raise SystemExit(
            "!! kaggle download failed. Accept the competition rules at "
            "https://www.kaggle.com/c/ieee-fraud-detection/rules first."
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def get_one(name: str) -> None:
    spec = DATASETS[name]
    if "files" in spec:
        print(f"[{name}]  {spec['detection']}  ·  {spec['license']}")
        for url, rel, sha, _size in spec["files"]:
            dest = RAW / rel
            if dest.exists():
                print(f"    already present: {dest.relative_to(ROOT)}")
            else:
                download(url, dest)
            verify(dest, sha)
        print()
        return
    out = RAW / spec["out"]
    print(f"[{name}]  {spec['detection']}  ·  {spec['license']}")
    if out.exists():
        print(f"    already present: {out.relative_to(ROOT)}")
    elif "kaggle_competition" in spec:
        fetch_kaggle(spec, out)
        # kaggle names the file <competition>.zip
        cand = out.parent / f"{spec['kaggle_competition']}.zip"
        if cand.exists():
            cand.replace(out)
    else:
        download(spec["url"], out)
    if out.exists():
        verify(out, spec.get("sha256"))
        if spec.get("unzip_to"):
            unzip(out, RAW / spec["unzip_to"])
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="PRAMAAN reproducible dataset downloader")
    ap.add_argument("datasets", nargs="*", help="dataset keys to fetch")
    ap.add_argument("--all", action="store_true", help="fetch every dataset")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    args = ap.parse_args()

    if args.list or (not args.datasets and not args.all):
        print("Available datasets:")
        for k, v in DATASETS.items():
            print(f"  {k:14s} {v['detection']:46s} {v['license']}")
        return

    names = list(DATASETS) if args.all else args.datasets
    for n in names:
        if n not in DATASETS:
            raise SystemExit(f"unknown dataset '{n}'. Use --list.")
        get_one(n)
    print("done.")


if __name__ == "__main__":
    main()
