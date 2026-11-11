from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

import tokenspeed_triton as triton
from tokenspeed_triton.runtime.build import compile_module_from_src

TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


def test_compile_module_bad_cache(fresh_knobs):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        fresh_knobs.cache.manager_class = InvalidFileCacheManager

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(Exception):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bundled toolchain discovery")
@pytest.mark.parametrize("arch", ["x64", "arm64"])
def test_cuda_bundled_editable_install(monkeypatch, tmp_path, arch):
    from tokenspeed_triton import windows_utils

    package = tmp_path / "triton"
    backend = package / "backends" / "nvidia"
    for relative in ("bin/ptxas.exe", "include/cuda.h", f"lib/{arch}/cuda.lib"):
        path = backend / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(windows_utils, "__file__", str(package / "windows_utils.py"))
    monkeypatch.setattr(windows_utils, "_arch_candidates", lambda: [arch])

    assert windows_utils.find_cuda_bundled() == (str(backend / "bin"), [str(backend / "include")],
                                                 [str(backend / "lib" / arch)])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bundled toolchain discovery")
def test_tcc_bundled_editable_install(monkeypatch, tmp_path):
    from tokenspeed_triton.runtime import build

    runtime = tmp_path / "triton" / "runtime"
    compiler = runtime / "tcc" / "tcc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setattr(build, "__file__", str(runtime / "build.py"))
    monkeypatch.setattr(build.sysconfig, "get_path", lambda name: str(tmp_path / "site-packages"))
    monkeypatch.setattr(build, "find_msvc_winsdk", lambda **kwargs: (None, [], []))
    monkeypatch.delenv("CC", raising=False)

    build._find_compiler.cache_clear()
    try:
        assert build._find_compiler("c") == str(compiler)
    finally:
        build._find_compiler.cache_clear()


@pytest.mark.parametrize("artifact", ["bootstrap", "codegen"])
@pytest.mark.parametrize(
    "mirror", ["", "https://mirror.example/generic/oaiartifacts", "https://mirror.example/generic/oaiartifacts/"])
def test_amd_codegen_download_mirror(artifact, mirror, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setenv("OAIARTIFACTS_BASE_URL", mirror)
    monkeypatch.delenv("TRITON_CACHE_DEPENDENCY_DOWNLOADS", raising=False)
    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(tmp_path))
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(tmp_path / "cache"),
            "--triton-llvm-system-suffix",
            "ubuntu-arm64",
            "--llvm-syspath",
            str(tmp_path / "unrelated-core-llvm"),
        ]))

    revision = "123456789abcdef"
    name = f"{'llvm' if artifact == 'bootstrap' else 'amd-codegen'}-{revision[:8]}-ubuntu-arm64-1"
    relative_path = "lib/cmake/llvm/LLVMConfig.cmake" if artifact == "bootstrap" else (
        f"lib/{build_helpers.amd_codegen_library_name()}")
    contents = b"test artifact"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{name}/{relative_path}")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    archive_bytes = buffer.getvalue()
    checksum = hashlib.sha256(archive_bytes).hexdigest()
    llvm_info = {"llvm_hash": revision, "build_number": 1, "sha256sum": {"ubuntu-arm64": checksum}}
    if artifact == "bootstrap":
        llvm_info["bootstrap_llvm"] = {"build_number": 1, "sha256sum": llvm_info.pop("sha256sum")}

    downloads = []

    def download(url, path, label):
        downloads.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(archive_bytes)

    monkeypatch.setattr(build_helpers, "_download_file", download)
    if artifact == "bootstrap":
        result = Path(build_helpers.download_codegen_llvm("amd", llvm_info, args)) / relative_path
    else:
        (tmp_path / "cmake").mkdir()
        (tmp_path / "cmake" / "amd-llvm-info.json").write_text(json.dumps(llvm_info))
        build_helpers.download_and_copy_amd_codegen(args)
        result = tmp_path / "third_party" / "amd" / "backend" / relative_path

    base = "https://mirror.example/generic/triton-llvm" if mirror else (
        "https://oaitriton.blob.core.windows.net/public/llvm-builds")
    assert downloads == [f"{base}/{name}.tar.gz"]
    assert result.read_bytes() == contents


def test_amd_codegen_bootstrap_build_flags(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setattr(build_helpers.sys, "platform", "linux")
    llvm_path = tmp_path / "llvm"
    for path in [llvm_path / "bin" / "clang", llvm_path / "bin" / "clang++"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    source_path = tmp_path / "third_party" / "amd" / "backend" / "codegen"
    source_path.mkdir(parents=True)
    devtoolset_root = tmp_path / "devtoolset"
    (devtoolset_root / "usr").mkdir(parents=True)

    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args(["--triton-cache-path", str(tmp_path / "cache")]))
    monkeypatch.setattr(build_helpers, "download_codegen_llvm", lambda *_: str(llvm_path))
    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(build_helpers, "get_llvm_system_suffix", lambda _: "almalinux-x64")
    commands = []
    monkeypatch.setattr(build_helpers.subprocess, "check_call", commands.append)
    monkeypatch.setenv("TRITON_BUILD_WITH_CLANG_LLD", "1")
    monkeypatch.setenv("DEVTOOLSET_ROOTPATH", str(devtoolset_root))

    build_helpers.build_amd_codegen({"llvm_hash": "123456789abcdef", "build_number": 1}, args)

    configure = commands[0]
    assert "-DCMAKE_LINKER=lld" in configure
    assert "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld" in configure
    assert "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld" in configure
    assert f"-DCMAKE_CXX_FLAGS=--gcc-toolchain={devtoolset_root / 'usr'}" in configure
    assert ("-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld -Wl,--gc-sections,--icf=safe "
            "-static-libstdc++ -static-libgcc") in configure
    assert os.path.join(str(llvm_path), "bin", "clang") in " ".join(configure)


@pytest.mark.parametrize("host, target, arch", [
    ("AMD64", "arm64", "arm64"),
    ("ARM64", "x64", "x64"),
    ("AMD64", None, "x64"),
])
def test_windows_toolchains_follow_build_target(monkeypatch, tmp_path, host, target, arch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setattr(build_helpers.platform, "system", lambda: "Windows")
    monkeypatch.setattr(build_helpers.platform, "machine", lambda: host)
    monkeypatch.setattr(build_helpers.sysconfig, "get_platform", lambda: "win-arm64" if host == "ARM64" else "win-amd64")
    if target is None:
        monkeypatch.delenv("VSCMD_ARG_TGT_ARCH", raising=False)
    else:
        monkeypatch.setenv("VSCMD_ARG_TGT_ARCH", target)
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(parser.parse_args(["--triton-cache-path", str(tmp_path)]))
    assert build_helpers.get_llvm_system_suffix(args) == f"windows-{arch}"
    wheel_arch = "arm64" if arch == "arm64" else "amd64"
    assert build_helpers._get_cmake_dir().name.startswith(f"cmake.win-{wheel_arch}-")

    downloads = []
    monkeypatch.setattr(build_helpers, "download_and_copy_amd_codegen", lambda _: None)
    monkeypatch.setattr(build_helpers, "download_and_copy", lambda **kwargs: downloads.append(kwargs))
    monkeypatch.setenv("TRITON_BUILD_PROTON", "OFF")
    monkeypatch.setenv("TRITON_BUILD_NVIDIA_GSAN_RUNTIME", "OFF")
    build_helpers.download_and_copy_dependencies(args)
    assert any(item["dst_path"].endswith(f"lib/{arch}/cuda.lib") for item in downloads)
    tinycc = next(item for item in downloads if item["dst_path"].endswith("/runtime/tcc"))
    assert tinycc["name"].endswith(f"-windows-{arch}")
