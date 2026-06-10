"""Canonical project paths so data/config locations don't depend on cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"

DATA.mkdir(exist_ok=True)
