"""Round-trip generated artifacts to a HuggingFace dataset repo.

All data (item bank, metadata, response matrices, frozen snapshots) lives on HF,
not in git — see configs/data.yaml for the repo pointer + pinned revision.

  push()          upload data/<artifacts> to the HF dataset repo
  pull()          download them back into data/ (at the pinned revision)
  load(name)      ensure one artifact is present locally, return its path
  create_repo()   create the dataset repo (you can also do this on the website)

Auth for pushing: `huggingface-cli login` or set HF_TOKEN in the env. Reads are
public if the repo is public, otherwise also need the token.

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


def _require_repo(cfg: dict | None = None) -> dict:
    cfg = cfg or _cfg()
    if not cfg.get("repo_id"):
        raise RuntimeError(
            f"No HF dataset configured. Set `repo_id` in {_CFG_PATH} "
            "(create the dataset repo on HuggingFace first).")
    return cfg


def create_repo(private: bool = True) -> str:
    """Create the configured dataset repo on HF (idempotent)."""
    from huggingface_hub import create_repo as _create
    cfg = _require_repo()
    _create(cfg["repo_id"], repo_type="dataset", private=private, exist_ok=True)
    return cfg["repo_id"]


def push(filename: str | None = None, message: str = "update artifacts") -> None:
    """Upload one artifact (or all configured) from data/ to the HF dataset."""
    from huggingface_hub import upload_file
    cfg = _require_repo()
    for name in ([filename] if filename else cfg["artifacts"]):
        path = DATA / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — generate it first")
        upload_file(path_or_fileobj=str(path), path_in_repo=name,
                    repo_id=cfg["repo_id"], repo_type="dataset",
                    commit_message=f"{message}: {name}")
        print(f"pushed {name} -> {cfg['repo_id']}")


def pull(filename: str | None = None, revision: str | None = None) -> list:
    """Download one artifact (or all configured) into data/ at the pinned revision."""
    from huggingface_hub import hf_hub_download
    cfg = _require_repo()
    rev = revision or cfg.get("revision")          # None -> latest on main
    out = []
    for name in ([filename] if filename else cfg["artifacts"]):
        local = hf_hub_download(cfg["repo_id"], name, repo_type="dataset", revision=rev)
        dest = DATA / name
        shutil.copy(local, dest)
        out.append(dest)
        print(f"pulled {name}@{rev or 'main'} -> {dest}")
    return out


def load(name: str, revision: str | None = None):
    """Ensure `name` exists under data/ (pull if missing); return its path."""
    dest = DATA / name
    if not dest.exists():
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
        print("HF data store:")
        print("  repo_id :", cfg.get("repo_id") or f"(unset — edit {_CFG_PATH})")
        print("  revision:", cfg.get("revision") or "latest (main)")
        print("  artifacts:", ", ".join(cfg.get("artifacts", [])))
        print("usage: python -m circuit_irt.datastore [status|push|pull]")
