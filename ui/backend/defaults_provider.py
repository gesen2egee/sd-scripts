"""Load training parser defaults for the ANIMA UI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT_DIR / "ui" / "runtime"
CACHE_PATH = RUNTIME_DIR / "defaults_cache.json"


def _to_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    return str(value)


def _extract_parser_defaults() -> Dict[str, Any]:
    venv_python = ROOT_DIR / "venv" / "Scripts" / "python.exe"
    python_exec = str(venv_python if venv_python.exists() else Path(sys.executable))

    script = r"""
import argparse
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

import anima_train_network as train_script

parser = train_script.setup_parser()
fields = {}

for action in parser._actions:
    if not action.option_strings:
        continue
    if action.dest in ("help",):
        continue

    option = None
    for opt in action.option_strings:
        if opt.startswith("--"):
            option = opt
            break
    if option is None:
        option = action.option_strings[0]

    choices = None
    if getattr(action, "choices", None) is not None:
        choices = list(action.choices)

    fields[action.dest] = {
        "dest": action.dest,
        "option": option,
        "default": action.default,
        "choices": choices,
        "help": action.help,
        "type": getattr(action.type, "__name__", str(action.type)) if getattr(action, "type", None) else None,
        "nargs": action.nargs,
    }

print(json.dumps({"fields": fields}, ensure_ascii=True))
""".strip()

    proc = subprocess.run(
        [python_exec, "-c", script, str(ROOT_DIR)],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "failed to extract parser defaults")

    raw = proc.stdout.strip()
    if not raw:
        raise RuntimeError("empty parser defaults output")
    # Keep the last non-empty line in case upstream logs are printed.
    last_line = raw.splitlines()[-1]
    parsed = json.loads(last_line)
    return _to_json_value(parsed)


def load_defaults() -> Tuple[Dict[str, Any], str | None]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    warning: str | None = None

    try:
        defaults = _extract_parser_defaults()
        defaults["source"] = "training_parser"
        CACHE_PATH.write_text(json.dumps(defaults, ensure_ascii=False, indent=2), encoding="utf-8")
        return defaults, warning
    except Exception as exc:  # pragma: no cover - exercised by fallback tests
        warning = f"Failed to load parser defaults: {exc}"
        if CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            cached["source"] = "cache"
            cached["warning"] = warning
            return cached, warning

        fallback = {
            "source": "fallback",
            "warning": warning,
            "fields": {},
        }
        return fallback, warning


def option_default(defaults: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    field = defaults.get("fields", {}).get(key, {})
    return field.get("default", fallback)

