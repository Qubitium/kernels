import io
import json
import platform
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import kernels.jit as jit_module
from kernels._rust import KernelDependency, KernelVersion, Oid
from kernels.jit import (
    FallbackResolver,
    JITBuildError,
    JITBuildInfoError,
    JITOptions,
    JITResolver,
    _cache_key,
    _commit_from_source_url,
    _is_cpu_variant,
    _load_cached_kernel,
    _normalize_source,
    _reject_unsupported_free_threaded_python,
    _resolve_source_commit,
    _safe_extract,
    _source_fingerprint,
    _source_subdir,
    _SourceSpec,
    _write_cache_entry,
)


def _cpu_variant_name() -> str:
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    cpu = platform.machine()
    os = platform.system().lower()
    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
    elif os == "windows":
        cpu = "x86_64" if cpu == "AMD64" else cpu
    return f"torch{major}{minor}-cpu-{cpu}-{os}"


def _make_cache_entry(tmp_path, *, validated_at: float | None = None):
    repo_id = "kernels-community/example"
    source = tmp_path / "source"
    source.mkdir()
    (source / "build.toml").write_text("version = 1\n")

    revision = Oid.from_str("a" * 40)
    source_spec = _SourceSpec(source=source, subdir=None, commit=None)
    cache_dir = tmp_path / "cache"
    key = _cache_key(repo_id=repo_id, revision=revision, source=source_spec)
    final_root = cache_dir / key
    variant = final_root / "build" / _cpu_variant_name()
    variant.mkdir(parents=True)
    (variant / "metadata.json").write_text(
        json.dumps(
            {
                "name": "example",
                "id": "example",
                "version": 1,
                "kernels-minver": "0.14.0",
                "license": "MIT",
                "python-depends": [],
                "kernel-depends": [],
                "backend": {"type": "cpu"},
            }
        )
    )

    _write_cache_entry(
        final_root=final_root,
        repo_id=repo_id,
        requested=KernelVersion.Version(1),
        revision=revision,
        source=source_spec,
        key=key,
    )
    if validated_at is not None:
        entry_path = final_root / "jit.json"
        payload = json.loads(entry_path.read_text())
        payload["validated_at"] = validated_at
        entry_path.write_text(json.dumps(payload))

    return repo_id, cache_dir, final_root, variant, source


class _MissingResolver:
    def resolve(self, *, api, backend, kernel):
        raise FileNotFoundError("no compatible build")


class _ResultResolver:
    def __init__(self, result):
        self.result = result

    def resolve(self, *, api, backend, kernel):
        return self.result


def test_fallback_resolver_uses_fallback_only_for_missing_build():
    fallback = _ResultResolver("fallback")
    resolver = FallbackResolver(primary=_MissingResolver(), fallback=fallback)
    assert resolver.resolve(api=None, backend="cpu", kernel=None) == "fallback"


def test_fallback_resolver_propagates_other_errors():
    class _BrokenResolver:
        def resolve(self, *, api, backend, kernel):
            raise ValueError("broken metadata")

    resolver = FallbackResolver(primary=_BrokenResolver(), fallback=_ResultResolver("fallback"))
    with pytest.raises(ValueError, match="broken metadata"):
        resolver.resolve(api=None, backend="cpu", kernel=None)


def test_fallback_resolver_uses_cached_result_before_primary():
    class _CachedResolver:
        def resolve_cached(self, *, backend, kernel):
            return "cached"

        def resolve(self, *, api, backend, kernel):
            raise AssertionError("fallback resolver should not run")

    class _PrimaryResolver:
        def resolve(self, *, api, backend, kernel):
            raise AssertionError("primary resolver should not run")

    resolver = FallbackResolver(primary=_PrimaryResolver(), fallback=_CachedResolver())
    assert resolver.resolve(api=None, backend="cpu", kernel=None) == "cached"


def test_normalize_github_tree_source():
    commit = "a" * 40
    source, subdir = _normalize_source(
        f"https://github.com/huggingface/kernels-community/tree/{commit}/quantization-gptq",
        None,
    )
    assert source == "https://github.com/huggingface/kernels-community"
    assert subdir == "quantization-gptq"
    assert (
        _commit_from_source_url(f"https://github.com/huggingface/kernels-community/tree/{commit}/quantization-gptq")
        == commit
    )


def test_normalize_local_source(tmp_path):
    source, subdir = _normalize_source(tmp_path, "kernel")
    assert source == tmp_path.resolve()
    assert subdir == "kernel"


def test_source_commit_requires_full_sha():
    assert _resolve_source_commit("a" * 40, None) == "a" * 40
    with pytest.raises(JITBuildError, match="40-character"):
        _resolve_source_commit("main", None)


def test_cpu_variant_filter_handles_arch_and_noarch_variants():
    cpu_arch = SimpleNamespace(arch=SimpleNamespace(backend=SimpleNamespace(name="cpu")))
    cuda_arch = SimpleNamespace(arch=SimpleNamespace(backend=SimpleNamespace(name="cuda")))
    cpu_noarch = SimpleNamespace(arch=SimpleNamespace(backend_name="cpu"))
    other_noarch = SimpleNamespace(arch=SimpleNamespace(backend_name="cuda"))

    assert _is_cpu_variant(cpu_arch)
    assert _is_cpu_variant(cpu_noarch)
    assert not _is_cpu_variant(cuda_arch)
    assert not _is_cpu_variant(other_noarch)


def test_source_subdir_rejects_traversal(tmp_path):
    with pytest.raises(JITBuildError, match="escapes"):
        _source_subdir(tmp_path, "../outside")


def test_safe_extract_rejects_traversal(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        data = b"escape"
        info = tarfile.TarInfo("../escape")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    payload.seek(0)

    with tarfile.open(fileobj=payload, mode="r:gz") as archive:
        with pytest.raises(JITBuildError, match="Unsafe path"):
            _safe_extract(archive, tmp_path)


def test_local_source_fingerprint_changes_with_source(tmp_path):
    source_file = tmp_path / "build.toml"
    source_file.write_text("version = 1\n")
    source = _SourceSpec(source=tmp_path, subdir=None, commit=None)
    first = _source_fingerprint(source)
    source_file.write_text("version = 2\n")
    assert _source_fingerprint(source) != first


def test_cache_key_is_stable_and_tracks_source(tmp_path):
    source_file = tmp_path / "build.toml"
    source_file.write_text("version = 1\n")
    source = _SourceSpec(source=tmp_path, subdir=None, commit=None)

    class _Revision:
        def __str__(self):
            return "b" * 40

    revision = _Revision()
    first = _cache_key(repo_id="kernels-community/example", revision=revision, source=source)
    assert _cache_key(repo_id="kernels-community/example", revision=revision, source=source) == first

    source_file.write_text("version = 2\n")
    assert _cache_key(repo_id="kernels-community/example", revision=revision, source=source) != first


def test_jit_options_are_explicit():
    assert JITOptions().source is None
    assert JITOptions(source=Path("/tmp/source")).source == Path("/tmp/source")


def test_jit_options_cache_ttl_defaults_to_one_hour():
    assert JITOptions().cache_ttl == 3600.0
    assert JITOptions(cache_ttl=None).cache_ttl is None


def test_cache_first_hit_within_ttl(tmp_path):
    repo_id, cache_dir, _, variant, _ = _make_cache_entry(tmp_path)
    resolver = JITResolver(options=JITOptions(cache_dir=cache_dir))
    kernel = KernelDependency(repo_id=repo_id, version=KernelVersion.Version(1))

    cached = resolver.resolve_cached(backend="cpu", kernel=kernel)

    assert cached is not None
    assert cached.variant_path == variant


def test_cache_first_expires_after_ttl(tmp_path):
    repo_id, cache_dir, _, _, _ = _make_cache_entry(tmp_path, validated_at=time.time() - 3601)

    expired = _load_cached_kernel(
        repo_id=repo_id,
        requested=KernelVersion.Version(1),
        options=JITOptions(cache_dir=cache_dir),
    )
    no_expiry = _load_cached_kernel(
        repo_id=repo_id,
        requested=KernelVersion.Version(1),
        options=JITOptions(cache_dir=cache_dir, cache_ttl=None),
    )
    longer_ttl = _load_cached_kernel(
        repo_id=repo_id,
        requested=KernelVersion.Version(1),
        options=JITOptions(cache_dir=cache_dir, cache_ttl=7200),
    )
    disabled = _load_cached_kernel(
        repo_id=repo_id,
        requested=KernelVersion.Version(1),
        options=JITOptions(cache_dir=cache_dir, cache_ttl=0),
    )

    assert expired is None
    assert no_expiry is not None
    assert longer_ttl is not None
    assert disabled is None


def test_cache_first_rejects_changed_local_source(tmp_path):
    repo_id, cache_dir, _, _, source = _make_cache_entry(tmp_path)
    (source / "build.toml").write_text("version = 2\n")

    cached = _load_cached_kernel(
        repo_id=repo_id,
        requested=KernelVersion.Version(1),
        options=JITOptions(cache_dir=cache_dir),
    )

    assert cached is None


def test_free_threaded_python_before_315_is_rejected(monkeypatch):
    monkeypatch.setattr(
        jit_module.sysconfig,
        "get_config_var",
        lambda name: 1 if name == "Py_GIL_DISABLED" else None,
    )
    monkeypatch.setattr(jit_module.sys, "version_info", (3, 14, 0))
    with pytest.raises(JITBuildInfoError, match="free-threaded"):
        _reject_unsupported_free_threaded_python()


def test_free_threaded_python_315_is_not_rejected(monkeypatch):
    monkeypatch.setattr(
        jit_module.sysconfig,
        "get_config_var",
        lambda name: 1 if name == "Py_GIL_DISABLED" else None,
    )
    monkeypatch.setattr(jit_module.sys, "version_info", (3, 15, 0))
    _reject_unsupported_free_threaded_python()
