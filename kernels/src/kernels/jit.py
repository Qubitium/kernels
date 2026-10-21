from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from huggingface_hub import HfApi, constants

from kernels._rust import KernelDependency, KernelVersion, Metadata, Oid
from kernels._versions import resolve_kernel_version
from kernels.backends import _select_backend
from kernels.hf_hub import _check_trust_remote_code, _get_cache_dir
from kernels.resolver import LocalKernel, RemoteKernel, Resolver
from kernels.variants import (
    get_variants,
    get_variants_local,
    parse_variant,
    resolve_variant,
    variants_trace_str,
)

logger = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GITHUB_KERNEL_SOURCE = "https://github.com/huggingface/kernels-community"
_JIT_CACHE_VERSION = "1"
_JIT_CACHE_ENTRY_VERSION = 1
_JIT_CACHE_ENTRY_FILE = "jit.json"
_IGNORE_SOURCE_PARTS = frozenset({"__pycache__", ".git", "build", "_cmake_build"})


class JITBuildError(RuntimeError):
    """Raised when a kernel cannot be compiled locally."""


@dataclass(frozen=True)
class JITOptions:
    """Options for compiling a kernel when no compatible prebuilt variant exists.

    The source must be reproducible. For Hub kernels this is normally the
    source repository and commit recorded in the build metadata. ``source``
    can override the recorded source for kernels whose Hub metadata predates
    source recording. ``cache_ttl`` controls how long a cached build can be
    reused without revalidating against the Hub.
    """

    source: str | Path | None = None
    source_subdir: str | None = None
    revision: str | None = None
    cache_dir: Path | None = None
    cache_ttl: float | None = 3600.0


@dataclass(frozen=True)
class _SourceSpec:
    source: str | Path
    subdir: str | None
    commit: str | None


@dataclass
class JITResolver:
    """Resolver that builds a CPU kernel when no compatible build exists."""

    options: JITOptions = field(default_factory=JITOptions)
    trust_remote_code: bool | list[str] = False

    def resolve_cached(self, *, backend: str | None, kernel: KernelDependency) -> LocalKernel | None:
        selected_backend = _select_backend(backend)
        if selected_backend.name != "cpu":
            return None
        return _load_cached_kernel(
            repo_id=kernel.repo_id,
            requested=kernel.version,
            options=self.options,
        )

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel:
        selected_backend = _select_backend(backend)
        if selected_backend.name != "cpu":
            raise JITBuildError(
                "The JIT fallback currently supports CPU kernels only; "
                f"the selected backend is `{selected_backend.name}`."
            )

        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )

        revision = resolve_kernel_version(
            kernel.repo_id,
            kernel.version,
            local_files_only=False,
        )
        candidates = _source_candidates(
            api=api,
            repo_id=kernel.repo_id,
            revision=revision,
            options=self.options,
        )
        if not candidates:
            raise JITBuildError(
                f"No JIT source is available for `{kernel.repo_id}`. "
                "Pass `jit=JITOptions(source=...)` with a pinned source tree "
                "or a GitHub repository URL."
            )

        errors: list[str] = []
        for candidate in candidates:
            try:
                return _build_local_kernel(
                    repo_id=kernel.repo_id,
                    revision=revision,
                    source=candidate,
                    options=self.options,
                    requested=kernel.version,
                )
            except Exception as exc:
                message = f"{candidate.source}@{candidate.commit or 'local'}: {exc}"
                errors.append(message)
                logger.warning("JIT build candidate failed: %s", message)

        raise JITBuildError(
            f"Could not compile a CPU kernel for `{kernel.repo_id}` from source. Tried:\n  - " + "\n  - ".join(errors)
        )


@dataclass
class FallbackResolver:
    """Prefer a cached fallback result, then use the fallback when the primary resolver has no compatible build."""

    primary: Resolver
    fallback: Resolver

    def resolve(
        self, *, api: HfApi, backend: str | None, kernel: KernelDependency
    ) -> LocalKernel | RemoteKernel | None:
        resolve_cached = getattr(self.fallback, "resolve_cached", None)
        if resolve_cached is not None:
            cached = resolve_cached(backend=backend, kernel=kernel)
            if cached is not None:
                return cached

        try:
            return self.primary.resolve(api=api, backend=backend, kernel=kernel)
        except FileNotFoundError:
            return self.fallback.resolve(api=api, backend=backend, kernel=kernel)


def _source_candidates(
    *,
    api: HfApi,
    repo_id: str,
    revision: Oid,
    options: JITOptions,
) -> list[_SourceSpec]:
    """Return source candidates in preferred order.

    The Hub metadata for a kernel can record the source repository. Older
    kernels do not have that field, so kernels in the ``kernels-community``
    organization fall back to the corresponding directory in the
    ``huggingface/kernels-community`` monorepo.
    """

    if options.source is not None:
        source_url_commit = _commit_from_source_url(options.source)
        commit = _resolve_source_commit(options.revision or source_url_commit, None)
        if commit is None and not _is_local_source(options.source):
            commit = _metadata_commit(
                api=api,
                repo_id=repo_id,
                revision=revision,
                source=options.source,
            )
        source, subdir = _normalize_source(options.source, options.source_subdir)
        return [_SourceSpec(source=source, subdir=subdir, commit=commit)]

    variants = [
        variant for variant in get_variants(api, repo_id=repo_id, revision=str(revision)) if _is_cpu_variant(variant)
    ]
    variants = sorted(variants, key=_variant_sort_key, reverse=True)

    candidates: list[_SourceSpec] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for variant in variants:
        try:
            metadata_path = Path(
                api.hf_hub_download(
                    repo_id,
                    repo_type="kernel",
                    filename=f"build/{variant.variant_str}/metadata.json",
                    cache_dir=_get_cache_dir(),
                    revision=str(revision),
                    local_files_only=False,
                )
            )
            metadata = Metadata.read_from_file(metadata_path)
        except Exception:
            continue

        source = metadata.source
        subdir = options.source_subdir
        if source is None and repo_id.startswith("kernels-community/"):
            source = _GITHUB_KERNEL_SOURCE
            subdir = subdir or str(metadata.name)
        if source is None:
            continue

        try:
            commit = _resolve_source_commit(
                options.revision or _commit_from_source_url(source),
                metadata.provenance.kernel.commit
                if metadata.provenance is not None and metadata.provenance.kernel is not None
                else None,
            )
            source, inferred_subdir = _normalize_source(source, subdir)
        except JITBuildError as exc:
            logger.warning(
                "Ignoring JIT source metadata for %s (%s): %s",
                repo_id,
                variant.variant_str,
                exc,
            )
            continue
        if not isinstance(source, Path) and commit is None:
            logger.warning(
                "Ignoring JIT source metadata for %s (%s): no pinned source commit",
                repo_id,
                variant.variant_str,
            )
            continue
        key = (str(source), inferred_subdir, commit)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_SourceSpec(source=source, subdir=inferred_subdir, commit=commit))

    return candidates


def _metadata_commit(
    *,
    api: HfApi,
    repo_id: str,
    revision: Oid,
    source: str | Path,
) -> str | None:
    """Find the source commit for an explicitly supplied source."""

    if _is_local_source(source):
        return None

    variants = sorted(
        (
            variant
            for variant in get_variants(api, repo_id=repo_id, revision=str(revision))
            if _is_cpu_variant(variant)
        ),
        key=_variant_sort_key,
        reverse=True,
    )
    for variant in variants:
        try:
            metadata_path = Path(
                api.hf_hub_download(
                    repo_id,
                    repo_type="kernel",
                    filename=f"build/{variant.variant_str}/metadata.json",
                    cache_dir=_get_cache_dir(),
                    revision=str(revision),
                    local_files_only=False,
                )
            )
            metadata = Metadata.read_from_file(metadata_path)
        except Exception:
            continue
        if metadata.provenance is not None and metadata.provenance.kernel is not None:
            return str(metadata.provenance.kernel.commit)
    return None


def _variant_sort_key(variant) -> tuple[int, ...]:
    """Prefer newer Torch variants when choosing source metadata."""

    framework = getattr(variant, "framework", None)
    version = getattr(framework, "version", None)
    if version is None:
        return (0,)
    try:
        return tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return (0,)


def _is_cpu_variant(variant) -> bool:
    """Return whether a variant targets the CPU backend."""

    backend = getattr(variant.arch, "backend", None)
    if backend is not None:
        return getattr(backend, "name", None) == "cpu"
    return getattr(variant.arch, "backend_name", None) == "cpu"


def _resolve_source_commit(explicit: str | None, recorded: str | None) -> str | None:
    if explicit is not None:
        if not _COMMIT_RE.fullmatch(explicit):
            raise JITBuildError(f"JIT source revisions must be full 40-character Git commit SHAs; got {explicit!r}.")
        return explicit
    if recorded is None:
        return None
    recorded = str(recorded)
    if not _COMMIT_RE.fullmatch(recorded):
        raise JITBuildError(f"Recorded source revision is not a full Git commit: {recorded!r}")
    return recorded


def _commit_from_source_url(source: str | Path) -> str | None:
    if _is_local_source(source):
        return None
    parts = [part for part in urllib.parse.urlsplit(str(source)).path.split("/") if part]
    if "tree" not in parts[2:]:
        return None
    tree_index = parts.index("tree", 2)
    if len(parts) <= tree_index + 1:
        return None
    candidate = parts[tree_index + 1]
    return candidate if _COMMIT_RE.fullmatch(candidate) else None


def _is_local_source(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    parsed = urllib.parse.urlsplit(str(source))
    return parsed.scheme == "" and Path(str(source)).expanduser().exists()


def _normalize_source(source: str | Path, subdir: str | None) -> tuple[str | Path, str | None]:
    if isinstance(source, Path) or _is_local_source(source):
        path = Path(source).expanduser().resolve()
        if not path.is_dir():
            raise JITBuildError(f"JIT source directory does not exist: {path}")
        return path, subdir

    value = str(source)
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise JITBuildError(f"JIT source must be a local directory or an http(s) Git repository URL; got {value!r}.")
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise JITBuildError(f"JIT source URLs currently must point at github.com; got {parsed.netloc!r}.")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise JITBuildError(f"Invalid GitHub source URL: {value!r}")
    repo = f"https://github.com/{path_parts[0]}/{path_parts[1].removesuffix('.git')}"

    if "tree" in path_parts[2:]:
        tree_index = path_parts.index("tree", 2)
        if len(path_parts) > tree_index + 2 and subdir is None:
            subdir = "/".join(path_parts[tree_index + 2 :])
    return repo, subdir


def _requested_cache_key(version: KernelVersion) -> dict[str, str | int]:
    if isinstance(version, KernelVersion.Version):
        return {"kind": "version", "value": version.version}
    if isinstance(version, KernelVersion.Revision):
        return {"kind": "revision", "value": version.revision}
    raise TypeError(f"Unsupported kernel version type: {type(version).__name__}")


def _write_cache_entry(
    *,
    final_root: Path,
    repo_id: str,
    requested: KernelVersion,
    revision: Oid,
    source: _SourceSpec,
    key: str,
) -> None:
    """Record the resolution metadata needed for a cache-first reload."""
    payload = {
        "version": _JIT_CACHE_ENTRY_VERSION,
        "repo_id": repo_id,
        "requested": _requested_cache_key(requested),
        "revision": str(revision),
        "source": str(source.source),
        "source_is_local": isinstance(source.source, Path),
        "source_subdir": source.subdir,
        "source_commit": source.commit,
        "key": key,
        "validated_at": time.time(),
    }
    entry_path = final_root / _JIT_CACHE_ENTRY_FILE
    fd, tmp_name = tempfile.mkstemp(
        dir=final_root,
        prefix=f".{_JIT_CACHE_ENTRY_FILE}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, sort_keys=True)
            tmp_file.write("\n")
        os.replace(tmp_name, entry_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _read_cache_entry(final_root: Path) -> dict | None:
    try:
        payload = json.loads((final_root / _JIT_CACHE_ENTRY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _JIT_CACHE_ENTRY_VERSION:
        return None
    return payload


def _cache_entry_is_fresh(payload: dict, ttl: float | None) -> bool:
    if ttl is None:
        return True
    if ttl <= 0:
        return False
    validated_at = payload.get("validated_at")
    if not isinstance(validated_at, (int, float)):
        return False
    return max(0.0, time.time() - validated_at) <= ttl


def _local_kernel_from_variant(*, repo_id: str, revision: Oid, variant: Path) -> LocalKernel:
    metadata = Metadata.read_from_file(variant / "metadata.json")
    origin = None
    try:
        origin = RemoteKernel(
            repo_id=repo_id,
            revision=revision,
            metadata=metadata,
            variant=parse_variant(variant.name),
        )
    except ValueError:
        # A locally generated variant name is still loadable; it just cannot be
        # represented as a remote build variant.
        pass
    return LocalKernel(variant_path=variant, metadata=metadata, origin=origin)


def _build_local_kernel(
    *,
    repo_id: str,
    revision: Oid,
    source: _SourceSpec,
    options: JITOptions,
    requested: KernelVersion,
) -> LocalKernel:
    _reject_unsupported_free_threaded_python()
    cache_root = _jit_cache_root(options.cache_dir)
    key = _cache_key(repo_id=repo_id, revision=revision, source=source)
    final_root = cache_root / key
    lock_path = cache_root / f"{key}.lock"

    with _exclusive_lock(lock_path):
        variant = _load_cached_variant(final_root)
        if variant is None:
            build_parent = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root))
            try:
                source_root = _materialize_source(source, build_parent)
                _run_build(source_root, source.commit)
                built_variant = _find_built_variant(source_root)
                final_root.mkdir(parents=True, exist_ok=True)
                final_variant = final_root / "build" / built_variant.name
                final_variant.parent.mkdir(parents=True, exist_ok=True)
                if final_variant.exists():
                    shutil.rmtree(final_variant)
                os.replace(built_variant, final_variant)
                variant = final_variant
            finally:
                shutil.rmtree(build_parent, ignore_errors=True)

    try:
        _write_cache_entry(
            final_root=final_root,
            repo_id=repo_id,
            requested=requested,
            revision=revision,
            source=source,
            key=key,
        )
    except OSError as exc:
        logger.warning("Could not write JIT cache metadata for %s: %s", final_root, exc)

    return _local_kernel_from_variant(repo_id=repo_id, revision=revision, variant=variant)


def _jit_cache_root(cache_dir: Path | None) -> Path:
    override = os.environ.get("KERNELS_JIT_CACHE")
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    elif override:
        root = Path(override).expanduser()
    else:
        root = Path(_get_cache_dir() or constants.HF_HUB_CACHE) / "jit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(*, repo_id: str, revision: Oid, source: _SourceSpec) -> str:
    import torch

    payload = {
        "version": _JIT_CACHE_VERSION,
        "repo_id": repo_id,
        "revision": str(revision),
        "source": str(source.source),
        "source_subdir": source.subdir,
        "source_commit": source.commit,
        "source_fingerprint": _source_fingerprint(source),
        "backend": "cpu",
        "torch": str(torch.__version__),
        "python": sys.version,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": {
            "builder": os.environ.get("KERNELS_JIT_BUILDER"),
            "cmake": os.environ.get("KERNELS_JIT_CMAKE"),
            "ninja": os.environ.get("KERNELS_JIT_NINJA"),
            "CC": os.environ.get("CC"),
            "CXX": os.environ.get("CXX"),
            "CFLAGS": os.environ.get("CFLAGS"),
            "CXXFLAGS": os.environ.get("CXXFLAGS"),
            "LDFLAGS": os.environ.get("LDFLAGS"),
            "CMAKE_ARGS": os.environ.get("CMAKE_ARGS"),
            "CMAKE_GENERATOR": os.environ.get("CMAKE_GENERATOR"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def _source_fingerprint(source: _SourceSpec) -> str | None:
    if not isinstance(source.source, Path):
        return None
    digest = hashlib.sha256()
    for path in sorted(source.source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source.source)
        if any(part in _IGNORE_SOURCE_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_cached_variant(final_root: Path) -> Path | None:
    build_root = final_root / "build"
    if not build_root.is_dir():
        return None
    variants = get_variants_local(build_root)
    variant, _ = resolve_variant(variants, "cpu")
    if variant is None:
        return None
    path = build_root / variant.variant_str
    return path if (path / "metadata.json").is_file() else None


def _load_cached_kernel(
    *,
    repo_id: str,
    requested: KernelVersion,
    options: JITOptions,
) -> LocalKernel | None:
    """Return a fresh local JIT build without contacting the Hub."""
    if options.cache_ttl is not None and options.cache_ttl <= 0:
        return None

    requested_key = _requested_cache_key(requested)
    normalized_source = None
    if options.source is not None:
        try:
            normalized_source = _normalize_source(options.source, options.source_subdir)
        except (JITBuildError, OSError):
            return None

    cache_root = _jit_cache_root(options.cache_dir)
    matches: list[tuple[float, Path, Oid]] = []
    try:
        entries = list(cache_root.iterdir())
    except OSError:
        return None

    for entry in entries:
        if not entry.is_dir():
            continue
        payload = _read_cache_entry(entry)
        if payload is None:
            continue
        if payload.get("repo_id") != repo_id or payload.get("requested") != requested_key:
            continue
        if not _cache_entry_is_fresh(payload, options.cache_ttl):
            continue

        if normalized_source is not None:
            if payload.get("source") != str(normalized_source[0]):
                continue
            if payload.get("source_subdir") != normalized_source[1]:
                continue
        elif options.source_subdir is not None and payload.get("source_subdir") != options.source_subdir:
            continue

        if options.revision is not None:
            if not _COMMIT_RE.fullmatch(options.revision):
                return None
            if payload.get("source_commit") != options.revision:
                continue

        revision_str = payload.get("revision")
        source_str = payload.get("source")
        source_subdir = payload.get("source_subdir")
        source_commit = payload.get("source_commit")
        source_is_local = payload.get("source_is_local")
        if not isinstance(revision_str, str) or not _COMMIT_RE.fullmatch(revision_str):
            continue
        if not isinstance(source_str, str) or not isinstance(source_is_local, bool):
            continue
        if source_subdir is not None and not isinstance(source_subdir, str):
            continue
        if source_commit is not None and (
            not isinstance(source_commit, str) or not _COMMIT_RE.fullmatch(source_commit)
        ):
            continue

        source = _SourceSpec(
            source=Path(source_str) if source_is_local else source_str,
            subdir=source_subdir,
            commit=source_commit,
        )
        try:
            key = _cache_key(
                repo_id=repo_id,
                revision=Oid.from_str(revision_str),
                source=source,
            )
        except (OSError, ValueError):
            continue
        if key != entry.name or payload.get("key") != key:
            continue

        variant = _load_cached_variant(entry)
        if variant is None:
            continue
        validated_at = payload.get("validated_at")
        matches.append(
            (
                float(validated_at) if isinstance(validated_at, (int, float)) else 0.0,
                variant,
                Oid.from_str(revision_str),
            )
        )

    if not matches:
        return None

    _, variant, revision = max(matches, key=lambda match: match[0])
    return _local_kernel_from_variant(repo_id=repo_id, revision=revision, variant=variant)


def _materialize_source(source: _SourceSpec, destination: Path) -> Path:
    if isinstance(source.source, Path):
        target = destination / "source"
        shutil.copytree(
            source.source,
            target,
            ignore=shutil.ignore_patterns(".git", "build", "_cmake_build", "__pycache__", "*.pyc"),
        )
        return _source_subdir(target, source.subdir)

    if source.commit is None:
        raise JITBuildError(
            "A pinned source commit is required for remote JIT builds. "
            "Pass `jit=JITOptions(revision='<40-character commit>')`."
        )

    parsed = urllib.parse.urlsplit(str(source.source))
    parts = [part for part in parsed.path.split("/") if part]
    owner, repo = parts[0], parts[1].removesuffix(".git")
    tarball = destination / "source.tar.gz"
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{source.commit}"
    request = urllib.request.Request(url, headers={"User-Agent": "kernels-jit"})
    with urllib.request.urlopen(request, timeout=60) as response, tarball.open("wb") as output:
        shutil.copyfileobj(response, output)

    extracted = destination / "extracted"
    extracted.mkdir()
    with tarfile.open(tarball, "r:gz") as archive:
        _safe_extract(archive, extracted)
    entries = [entry for entry in extracted.iterdir() if entry.is_dir()]
    if len(entries) != 1:
        raise JITBuildError(f"Unexpected source archive layout for {url}")
    return _source_subdir(entries[0], source.subdir)


def _source_subdir(root: Path, subdir: str | None) -> Path:
    if subdir is None:
        return root
    path = (root / subdir).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise JITBuildError(f"JIT source subdirectory escapes the source tree: {subdir!r}") from exc
    if not path.is_dir():
        raise JITBuildError(f"JIT source subdirectory does not exist: {path}")
    return path


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        try:
            member_path.relative_to(root)
        except ValueError as exc:
            raise JITBuildError(f"Unsafe path in source archive: {member.name!r}") from exc
        if not (member.isdir() or member.isreg()):
            raise JITBuildError(f"Unsafe entry in source archive: {member.name!r}")
    archive.extractall(destination)


def _run_build(source_root: Path, commit: str | None) -> None:
    builder = _find_tool("KERNELS_JIT_BUILDER", "kernel-builder")
    cmake = _find_tool("KERNELS_JIT_CMAKE", "cmake")
    ninja = _find_tool("KERNELS_JIT_NINJA", "ninja", required=False)
    if builder is None:
        raise JITBuildError(
            "`kernel-builder` is required for JIT builds. Install it with "
            "`pip install hf-kernel-builder` or set KERNELS_JIT_BUILDER."
        )
    if cmake is None:
        raise JITBuildInfoError("`cmake` is required for JIT builds; install cmake >= 3.26.")

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [
            str(Path(sys.executable).parent),
            str(Path(builder).parent),
            env.get("PATH", ""),
        ]
    )
    if ninja is not None:
        env.setdefault("CMAKE_GENERATOR", "Ninja")
        env.setdefault("CMAKE_MAKE_PROGRAM", str(ninja))
    env.setdefault("MAX_JOBS", str(max(1, os.cpu_count() or 1)))

    command = [str(builder), "create-pyproject", "--force"]
    if commit is not None:
        command.extend(["--kernel-sha", commit])
    command.append(str(source_root))
    _run_command(command, cwd=source_root, env=env, description="generate kernel build files")

    _run_command(
        [sys.executable, "setup.py", "build_kernel"],
        cwd=source_root,
        env=env,
        description="compile CPU kernel",
    )


def _find_tool(env_name: str, name: str, *, required: bool = True) -> str | None:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise JITBuildError(f"{env_name} does not point to a file: {path}")
        return str(path)
    path = shutil.which(name)
    if path is None:
        venv_path = Path(sys.executable).parent / name
        if venv_path.is_file():
            path = str(venv_path)
    if path is None and required:
        return None
    return path


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    description: str,
) -> None:
    logger.info("JIT build: %s (%s)", description, " ".join(command))
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise JITBuildInfoError(f"Could not run {command[0]}: {exc}") from exc
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise JITBuildError(
            f"JIT build step failed ({description}, exit {result.returncode})." + (f"\n{tail}" if tail else "")
        )


class JITBuildInfoError(JITBuildError):
    """A JIT build failure caused by missing local build prerequisites."""


def _find_built_variant(source_root: Path):
    build_root = source_root / "build"
    variants = get_variants_local(build_root)
    variant, trace = resolve_variant(variants, "cpu")
    if variant is None:
        raise JITBuildInfoError(
            "The JIT build did not produce a CPU variant compatible with this "
            f"environment:\n\n{variants_trace_str(trace)}"
        )
    return build_root / variant.variant_str


def _reject_unsupported_free_threaded_python() -> None:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    if sys.version_info < (3, 15):
        raise JITBuildInfoError(
            "JIT kernel builds are disabled on free-threaded Python < 3.15. "
            "The current HF kernel C++ ABI is not compatible with this interpreter. "
            "Use a GIL-enabled Python build or a newer free-threaded ABI."
        )


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "FallbackResolver",
    "JITBuildError",
    "JITBuildInfoError",
    "JITOptions",
    "JITResolver",
]
