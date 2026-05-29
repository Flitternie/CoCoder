from __future__ import annotations

from pathlib import Path
import json

from .config import DATASET_ROOT
from .models import RepoBundle


def list_repos(dataset: str) -> list[str]:
    ds_dir = DATASET_ROOT / dataset
    if not ds_dir.exists():
        raise ValueError(f"Dataset does not exist: {dataset}")
        # Dataset not found: {dataset}

    repos: list[str] = []
    for item in ds_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith("."):
            continue
        repos.append(item.name)
    repos.sort(key=str.lower)
    return repos


def get_numbered_repo_name(dataset: str, repo: str) -> str:
    """Return repo name with a numbered prefix, e.g. '11-python-hl7'.
    Numbering based on alphabetical order from list_repos()."""
    repos = list_repos(dataset)
    try:
        idx = repos.index(repo)
    except ValueError:
        # repo not in dataset, return as-is
        return repo
    return f"{idx + 1:02d}-{repo}"


def list_repos_numbered(dataset: str) -> list[str]:
    """Return all repos with numbered prefixes."""
    repos = list_repos(dataset)
    return [f"{i + 1:02d}-{name}" for i, name in enumerate(repos)]


def _read_text(base: Path, rel: str | None) -> str:
    if not rel:
        return ""
    path = base / rel
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_repo_bundle(dataset: str, repo: str) -> RepoBundle:
    repo_dir = DATASET_ROOT / dataset / repo
    if not repo_dir.exists():
        raise ValueError(f"Repository does not exist: {dataset}/{repo}")
        # Repository not found: {dataset}/{repo}

    config_path = repo_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json: {config_path}")
        # Missing config.json: {config_path}

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    requirement = _read_text(repo_dir, cfg.get("PRD"))

    uml_class = ""
    uml_sequence = ""

    if dataset == "DevEval":
        uml_class = _read_text(repo_dir, cfg.get("UML_class"))
        uml_sequence = _read_text(repo_dir, cfg.get("UML_sequence"))
    elif dataset == "CodeProjectEval":
        uml_paths = cfg.get("UML", [])
        chosen: str | None = None
        for item in uml_paths:
            if "pyreverse" in item:
                chosen = item
                break
        if chosen is None and uml_paths:
            chosen = uml_paths[0]
        uml_class = _read_text(repo_dir, chosen)
        uml_sequence = ""

    arch_design = _read_text(repo_dir, cfg.get("architecture_design"))

    return RepoBundle(
        dataset=dataset,
        repo=repo,
        repo_dir=str(repo_dir),
        config=cfg,
        requirement=requirement,
        uml_class=uml_class,
        uml_sequence=uml_sequence,
        arch_design=arch_design,
    )
