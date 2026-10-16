"""Resolve workflow paths from portable defaults and optional local overrides."""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def configured_paths():
    """Load defaults, then apply paths from KELP_CONFIG or config.json."""
    defaults = json.loads((REPO_ROOT / "example_config.json").read_text())["paths"]
    override = Path(os.environ.get("KELP_CONFIG", REPO_ROOT / "config.json")).expanduser()
    if not override.is_absolute():
        override = REPO_ROOT / override
    if override.exists():
        supplied = json.loads(override.read_text())["paths"]
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown path settings in {override.name}: {sorted(unknown)}")
        defaults.update(supplied)
    elif "KELP_CONFIG" in os.environ:
        raise FileNotFoundError(override)
    return defaults


def get_path(name):
    """Resolve one path lazily; reject missing values and reference cycles."""
    paths = configured_paths()
    active = set()

    class Values(dict):
        def __missing__(self, key):
            if key in active:
                raise ValueError(f"Circular path reference: {key}")
            value = paths.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Set paths.{key} in your configuration before this workflow.")
            active.add(key)
            try:
                value = value.format_map(self)
            finally:
                active.remove(key)
            self[key] = value
            return value

    value = Path(Values(REPO_ROOT=str(REPO_ROOT))[name]).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def get_service(name):
    """Return a named service endpoint from configs/services.json."""
    return json.loads((REPO_ROOT / "configs/services.json").read_text())[name]


def __getattr__(name):
    if name in json.loads((REPO_ROOT / "example_config.json").read_text())["paths"]:
        return get_path(name)
    raise AttributeError(name)
