"""Predictable on-disk model storage for Chatterbox.

Downloads Hugging Face snapshots into a dedicated cache directory instead of
mutating the Hub cache or CWD-relative folders like ``./t3-model``. Local
checkpoints are used as-is from the path you pass.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Iterable, Optional

REPO_ID = "ResembleAI/chatterbox"

# Cache layout: {cache_dir}/{variant}/{revision}/
CACHE_ENV_VAR = "CHATTERBOX_MODEL_DIR"


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    revision: str
    t3_weights: str
    tokenizer: str
    files: tuple[str, ...]


ENGLISH_SPEC = ModelSpec(
    variant="english",
    revision="1b475dffa71fb191cb6d5901215eb6f55635a9b6",
    t3_weights="t3_cfg.safetensors",
    tokenizer="tokenizer.json",
    files=(
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ),
)

MULTILINGUAL_SPEC = ModelSpec(
    variant="multilingual",
    revision="05e904af2b5c7f8e482687a9d7336c5c824467d9",
    t3_weights="t3_mtl23ls_v2.safetensors",
    tokenizer="grapheme_mtl_merged_expanded_v1.json",
    files=(
        "ve.safetensors",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.safetensors",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ),
)

SPECS: dict[str, ModelSpec] = {
    ENGLISH_SPEC.variant: ENGLISH_SPEC,
    MULTILINGUAL_SPEC.variant: MULTILINGUAL_SPEC,
}


@dataclass(frozen=True)
class ResolvedModel:
    """Locations used to load a fully prepared checkpoint."""

    asset_dir: Path
    vllm_dir: Path
    t3_weights: Path
    spec: ModelSpec

    @property
    def variant(self) -> str:
        return self.spec.variant


def get_spec(variant: str) -> ModelSpec:
    try:
        return SPECS[variant]
    except KeyError as exc:
        supported = ", ".join(SPECS)
        raise ValueError(f"Unknown model variant {variant!r}. Expected one of: {supported}") from exc


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "chatterbox-vllm").resolve()


def variant_dir(variant: str, revision: Optional[str] = None, cache_dir: Optional[str | Path] = None) -> Path:
    spec = get_spec(variant)
    root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else default_cache_dir()
    return root / spec.variant / (revision or spec.revision)


def missing_files(directory: Path, filenames: Iterable[str]) -> list[str]:
    return [name for name in filenames if not (directory / name).is_file()]


def _is_writable_dir(directory: Path) -> bool:
    if not directory.exists():
        return False
    probe = directory / f".chatterbox-write-{os.getpid()}"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    try:
        if src.parent == dest.parent.resolve():
            dest.symlink_to(src.name)
        else:
            dest.symlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def _write_vllm_config(dest: Path) -> None:
    # Read via the top-level package so we don't import models.t3.__init__
    # (that module registers vLLM model classes).
    dest.write_bytes(
        pkg_files("chatterbox_vllm").joinpath("models", "t3", "config.json").read_bytes()
    )


def resolve_t3_weights(asset_dir: Path, spec: ModelSpec) -> Path:
    named = asset_dir / spec.t3_weights
    generic = asset_dir / "model.safetensors"
    if named.is_file():
        return named
    if generic.is_file():
        return generic
    raise FileNotFoundError(
        f"Could not find T3 weights in {asset_dir}. "
        f"Expected {spec.t3_weights!r} or 'model.safetensors'."
    )


def _overlay_dir(asset_dir: Path) -> Path:
    digest = hashlib.sha256(str(asset_dir.resolve()).encode()).hexdigest()[:16]
    return default_cache_dir() / "overlays" / digest


def prepare_vllm_dir(asset_dir: Path, spec: ModelSpec) -> Path:
    """Ensure vLLM can load from a directory containing config.json + model.safetensors.

    Prefers the checkpoint directory itself. If that path is read-only (common
    with Docker volume mounts), a small overlay is created in the cache.
    """
    t3_weights = resolve_t3_weights(asset_dir, spec)
    has_config = (asset_dir / "config.json").is_file()
    has_model = (asset_dir / "model.safetensors").is_file()
    if has_config and has_model:
        return asset_dir

    target = asset_dir if _is_writable_dir(asset_dir) else _overlay_dir(asset_dir)
    if target != asset_dir:
        print(
            f"Model directory {asset_dir} is not writable; "
            f"preparing vLLM files in overlay {target}"
        )
    target.mkdir(parents=True, exist_ok=True)

    config_path = target / "config.json"
    if not config_path.is_file():
        _write_vllm_config(config_path)

    model_path = target / "model.safetensors"
    if model_path.is_symlink() and not model_path.exists():
        model_path.unlink()
    if not model_path.is_file():
        _link_or_copy(t3_weights, model_path)

    return target


def _validate_asset_dir(asset_dir: Path, spec: ModelSpec) -> None:
    required = [name for name in spec.files if name != spec.t3_weights]
    missing = missing_files(asset_dir, required)
    try:
        resolve_t3_weights(asset_dir, spec)
    except FileNotFoundError:
        missing.append(spec.t3_weights)
    if missing:
        listed = ", ".join(spec.files)
        raise FileNotFoundError(
            f"Incomplete {spec.variant} model directory {asset_dir}. "
            f"Missing: {', '.join(missing)}. Expected files: {listed}."
        )


def resolve_local_model(ckpt_dir: str | Path, variant: str) -> ResolvedModel:
    """Resolve a local checkpoint directory. Never consults CWD-relative t3-model folders."""
    spec = get_spec(variant)
    asset_dir = Path(ckpt_dir).expanduser().resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {asset_dir}")
    _validate_asset_dir(asset_dir, spec)
    vllm_dir = prepare_vllm_dir(asset_dir, spec)
    return ResolvedModel(
        asset_dir=asset_dir,
        vllm_dir=vllm_dir,
        t3_weights=resolve_t3_weights(asset_dir, spec),
        spec=spec,
    )


def download_model(
    variant: str = "english",
    *,
    repo_id: str = REPO_ID,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
) -> Path:
    """Download a variant into a stable local directory and return that path.

    Files land in ``{cache_dir}/{variant}/{revision}/`` unless ``local_dir`` is
    given, in which case they are placed directly there (useful for Docker
    image builds).
    """
    spec = get_spec(variant)
    revision = revision or spec.revision
    dest = Path(local_dir).expanduser().resolve() if local_dir is not None else variant_dir(variant, revision, cache_dir)
    dest.mkdir(parents=True, exist_ok=True)

    missing = missing_files(dest, spec.files)
    try:
        from huggingface_hub.constants import HF_HUB_OFFLINE as _HF_OFFLINE
    except ImportError:
        _HF_OFFLINE = False
    offline = local_files_only or _HF_OFFLINE
    if missing and offline:
        raise FileNotFoundError(
            f"Model files missing in {dest} and downloads are disabled "
            f"(local_files_only/HF_HUB_OFFLINE): {', '.join(missing)}"
        )
    if missing:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {len(missing)} {spec.variant} model file(s) to {dest}")
        for filename in missing:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(dest),
            )

    # Ensure vLLM sidecar files exist in the downloaded tree when writable.
    prepare_vllm_dir(dest, spec)
    return dest


def configure_tokenizers(model_dir: Path, variant: str) -> None:
    """Point custom vLLM tokenizers at files in ``model_dir`` before LLM construction."""
    if variant == "english":
        from chatterbox_vllm.models.t3.entokenizer import EnTokenizer
        EnTokenizer.set_model_dir(model_dir)
    else:
        from chatterbox_vllm.models.t3.mtltokenizer import MTLTokenizer
        MTLTokenizer.set_model_dir(model_dir)


def looks_like_local_dir(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    return candidate.is_dir()
