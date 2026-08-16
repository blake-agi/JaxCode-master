"""Path resolution shared by the scripts: the PyTorch original, and the
working notebooks directory.

## Where the PyTorch original lives — one definition for all three gates.

check_alignment.py, check_signatures.py and build_study.py all need the
TorchCode original next to this checkout, and each used to hardcode the same
sibling path. Two layouts are in the wild:

    <parent>/JAXCode/  +  <parent>/TorchCode-master-original/   (zip names)
    <parent>/jaxcode/  +  <parent>/upstream/torchcode/          (by role)

Both are probed, and TORCHCODE_ORIGINAL overrides either. A missing original is
not an error — every caller degrades to skipping its check, because CI clones
this repo alone.

## Where the working notebooks live.

JAXCODE_NOTEBOOKS_DIR points at them when they are kept outside this checkout
(a private practice repo); unset, it is ROOT/notebooks as in any fresh clone.
That fallback is the trap: it is indistinguishable from a fresh clone, so a
shell that never loaded .env silently resolves to a path that does not exist.
VS Code loads .env for its terminal and Jupyter kernel via python.envFile; a
plain shell does not.

notebooks_dir_problem() diagnoses that, so no caller has to guess. What each
caller does with the diagnosis differs, and the difference matters:

    list_edited_notebooks.py  fatal — empty output means "nothing was edited",
                              so it must never also mean "wrong directory"
    refresh_notebooks.py      fatal — it CREATES the directory, so a wrong path
                              materializes a phantom tree of 58 templates
    build_study.py            warn — only used for "does this notebook exist"
                              links, and CI legitimately has no notebooks
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CANDIDATES = (
    ROOT.parent / "upstream" / "torchcode",
    ROOT.parent / "TorchCode-master-original",
)


def original_repo() -> Path:
    """Path to the PyTorch original. May not exist — callers check."""
    env = os.environ.get("TORCHCODE_ORIGINAL")
    if env:
        return Path(env)
    for candidate in _CANDIDATES:
        if candidate.exists():
            return candidate
    return _CANDIDATES[0]


NOTEBOOKS_ENV = "JAXCODE_NOTEBOOKS_DIR"


def notebooks_dir() -> Path:
    """The working notebooks directory. May not exist — see the guard below."""
    return Path(os.environ.get(NOTEBOOKS_ENV) or ROOT / "notebooks")


def _unloaded_env_file() -> Path | None:
    """A .env beside this repo that sets JAXCODE_NOTEBOOKS_DIR while the actual
    environment does not — i.e. it exists and was never loaded.

    This is what separates "you forgot to source .env" from "fresh clone, run
    make notebooks". Both leave the variable unset; only one has a file sitting
    there declaring where the notebooks really are.
    """
    if os.environ.get(NOTEBOOKS_ENV):
        return None
    for env_file in (ROOT / ".env", ROOT.parent / ".env"):
        try:
            if env_file.is_file() and NOTEBOOKS_ENV in env_file.read_text():
                return env_file
        except OSError:
            continue
    return None


def notebooks_dir_problem(*, require_exists: bool = True,
                          require_pristine: bool = False) -> str | None:
    """Diagnose an unusable notebooks directory. None when everything is fine.

    Returns a ready-to-print multi-line message (no color; callers style it).

    The unloaded-.env check always runs — it is the root cause, and no caller
    ever wants it. The other two are opt-in because "the directory is not there"
    is a legitimate state for some callers and not others:

        require_exists    off for refresh_notebooks.py, which creates it; a
                          fresh clone genuinely has no notebooks/ yet. Off too
                          for build_study.py, where CI has none and the only
                          cost is omitted links.
        require_pristine  on only for callers that diff against the baseline,
                          where its absence silently skips every notebook.
    """
    work = notebooks_dir()
    unloaded = _unloaded_env_file()

    if unloaded is not None:
        return (f"{NOTEBOOKS_ENV} is unset, so this fell back to {work}\n"
                f"  but {unloaded} sets it — that env file was never loaded.\n"
                f"  Load it and re-run:\n"
                f"      set -a && . {unloaded} && set +a")

    if require_exists and not work.is_dir():
        source = (f"{NOTEBOOKS_ENV}={os.environ[NOTEBOOKS_ENV]}"
                  if os.environ.get(NOTEBOOKS_ENV)
                  else f"the in-repo default ({NOTEBOOKS_ENV} is unset)")
        return (f"notebooks directory does not exist: {work}\n"
                f"  from: {source}\n"
                f"  Fresh clone? Generate them:  make notebooks")

    if require_pristine and not (work / "_pristine").is_dir():
        return (f"pristine baseline missing: {work / '_pristine'}\n"
                f"  Every notebook would be skipped as unjudgeable, printing "
                f"nothing —\n"
                f"  indistinguishable from 'nothing was edited'.\n"
                f"  Regenerate it:  make notebooks")

    return None
