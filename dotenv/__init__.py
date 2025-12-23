"""Minimal local shim for `dotenv.load_dotenv`.

This provides a tiny, dependency-free implementation of `load_dotenv` sufficient
for this repository: it reads a `.env`-style file and sets environment variables
in `os.environ` (optionally overriding existing values).

It intentionally does NOT implement the full python-dotenv API; it just provides
what the repo needs to avoid ModuleNotFoundError while keeping behavior useful
for local development.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional


def _parse_env_lines(lines: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Strip simple quotes
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        result[key] = val
    return result


def load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """Load environment variables from a `.env` file.

    Args:
      dotenv_path: Path to dotenv file. If None, will look for a file named `.env`
        in the current working directory.
      override: If True, existing environment variables will be overwritten.

    Returns:
      True if a dotenv file was found and parsed, False otherwise.
    """
    if dotenv_path is None:
        dotenv_path = os.path.join(os.getcwd(), ".env")

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return False

    parsed = _parse_env_lines(lines)
    for k, v in parsed.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return True


def dotenv_values(dotenv_path: Optional[str] = None) -> Dict[str, str]:
    """Return the key/value pairs that would be loaded from the given dotenv path."""
    if dotenv_path is None:
        dotenv_path = os.path.join(os.getcwd(), ".env")
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            return _parse_env_lines(f.readlines())
    except FileNotFoundError:
        return {}


__all__ = ["load_dotenv", "dotenv_values"]
