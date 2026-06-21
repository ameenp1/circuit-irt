"""Access generated artifacts (item bank, metadata, response matrices, frozen
snapshots). Backend is set in configs/data.yaml:

  backend: local   -> artifacts live in the data/ folder (current default).
                      push/pull are no-ops; data is produced by the generators
                      (e.g. `python -m circuit_irt.reference`).
  backend: hf      -> round-trip data/ <-> a HuggingFace dataset repo, with a
                      pinned revision for reproducibility.

  load(name)   ensure an artifact is present under data/, return its path
  push()/pull() sync with HF (only when backend: hf)
  create_repo() create the HF dataset repo

The library modules read data/ directly via circuit_irt.paths.DATA, so the
local pipeline works with no datastore/HF dependency. Auth for HF pushing:
`huggingface-cli login` or HF_TOKEN in the env.

CLI:  python -m circuit_irt.datastore [status|push|pull]
"""
from __future__ import annotations

import shutil
import sys

import yaml

from circuit_irt.paths import DATA, CONFIGS

_CFG_PATH = CONFIGS / "data.yaml"


def _cfg() -> dict:
    return yaml.safe_load(open(_CFG_PATH))


def _backend(cfg: dict | None = None) -> str:
    return (cfg or _cfg()).get("backend", "local")


def _require_hf(cfg: dict) -> dict:
    hf = cfg.get("hf") or {}
    if _backend(cfg) != "hf" or not hf.get("repo_id"):
        raise RuntimeError(
            f"HF backend not configured. In {_CFG_PATH} set `backend: hf` and "
            "`hf.repo_id` (create the dataset repo on HuggingFace first).")
    return hf


def create_repo(private: bool = True) -> str:
    from huggingface_hub import create_repo as _create
    hf = _require_hf(_cfg())
    _create(hf["repo_id"], repo_type="dataset", private=private, exist_ok=True)
    return hf["repo_id"]


def push(filename: str | None = None, message: str = "update artifacts") -> None:
    cfg = _cfg()
    if _backend(cfg) == "local":
        print("backend=local: data lives in data/ — nothing to sync. "
              "Set `backend: hf` in configs/data.yaml to enable HF push.")
        return
    from huggingface_hub import upload_file
    hf = _require_hf(cfg)
    for name in ([filename] if filename else cfg["artifacts"]):
        path = DATA / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — generate it first")
        upload_file(path_or_fileobj=str(path), path_in_repo=name,
                    repo_id=hf["repo_id"], repo_type="dataset",
                    commit_message=f"{message}: {name}")
        print(f"pushed {name} -> {hf['repo_id']}")


def pull(filename: str | None = None, revision: str | None = None) -> list:
    cfg = _cfg()
    if _backend(cfg) == "local":
        print("backend=local: data already lives in data/ — nothing to pull.")
        return [DATA / n for n in ([filename] if filename else cfg["artifacts"])]
    from huggingface_hub import hf_hub_download
    hf = _require_hf(cfg)
    rev = revision or hf.get("revision")
    out = []
    for name in ([filename] if filename else cfg["artifacts"]):
        local = hf_hub_download(hf["repo_id"], name, repo_type="dataset", revision=rev)
        dest = DATA / name
        shutil.copy(local, dest)
        out.append(dest)
        print(f"pulled {name}@{rev or 'main'} -> {dest}")
    return out


def load(name: str, revision: str | None = None):
    """Ensure `name` exists under data/ (pull from HF if missing); return its path."""
    dest = DATA / name
    if dest.exists():
        return dest
    if _backend() == "local":
        raise FileNotFoundError(
            f"{dest} not found. Generate it (e.g. `python -m circuit_irt.reference`).")
    pull(name, revision)
    return dest


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "push":
        push()
    elif cmd == "pull":
        pull()
    else:
        cfg = _cfg()
        print("data store:")
        print("  backend  :", _backend(cfg))
        if _backend(cfg) == "local":
            print("  location :", DATA)
        else:
            hf = cfg.get("hf") or {}
            print("  repo_id  :", hf.get("repo_id") or "(unset)")
            print("  revision :", hf.get("revision") or "latest (main)")
        print("  artifacts:", ", ".join(cfg.get("artifacts", [])))
        print("usage: python -m circuit_irt.datastore [status|push|pull]")
