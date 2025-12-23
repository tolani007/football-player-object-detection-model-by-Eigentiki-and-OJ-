import os
import tempfile
import pathlib

from dotenv import load_dotenv, dotenv_values


def test_load_dotenv_creates_env(tmp_path):
    p = tmp_path / "test.env"
    p.write_text("FOO=bar\n# comment\nBAZ='qux'\n")

    # ensure not present beforehand
    os.environ.pop("FOO", None)
    os.environ.pop("BAZ", None)

    assert load_dotenv(str(p)) is True
    assert os.environ.get("FOO") == "bar"
    assert os.environ.get("BAZ") == "qux"


def test_load_dotenv_override(tmp_path):
    p = tmp_path / "test2.env"
    p.write_text("VAL=fromfile\n")

    os.environ["VAL"] = "original"
    assert load_dotenv(str(p), override=False) is True
    assert os.environ["VAL"] == "original"

    assert load_dotenv(str(p), override=True) is True
    assert os.environ["VAL"] == "fromfile"


def test_dotenv_values_missing(tmp_path):
    p = tmp_path / "none.env"
    if p.exists():
        p.unlink()
    assert dotenv_values(str(p)) == {}
