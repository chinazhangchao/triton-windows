============
Installation
============

For supported platform/OS and supported hardware, review the `Compatibility <https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility>`_ section on Github.

--------------------
Binary Distributions
--------------------

You can install the latest stable release of Triton from pip:

.. code-block:: bash

      pip install triton

Binary wheels are available for CPython 3.10-3.14.

-----------
From Source
-----------

++++++++++++++
Python Package
++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git
      cd triton

      pip install -r python/requirements.txt # build-time dependencies
      pip install -e .

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

On Windows, use a Visual Studio developer shell with the C++ build tools and
Windows SDK installed. The build downloads the pinned Windows LLVM and NVIDIA
toolchain packages and bundles TinyCC for compiling runtime launchers. Both
x64 and ARM64 builds are supported. The default distribution name is
``tokenspeed-triton-windows``; the Python import is ``tokenspeed_triton``.
It can coexist with the canonical ``triton`` package, but must not be installed
alongside ``tokenspeed-triton``, which owns the same import namespace.

After installing the build-time dependencies, build from that shell with:

.. code-block:: powershell

      python -m pip install -e . --no-build-isolation

For release wheels, use Python 3.11 for the version-specific wheel and Python
3.12 with ``TRITON_STABLE_ABI=ON`` for regular CPython 3.12 through 3.14.
Set ``TRITON_BUILD_PROTON=OFF``, ``TRITON_EXT_ENABLED=OFF`` and
``TRITON_BUILD_UT=OFF``, then run ``python -m build --wheel``. Proton is a
separate distribution and is not included in this wheel.
Python 3.15 is excluded because its headers are incompatible with the bundled
TinyCC compiler.

To cross-compile ARM64 wheels on Windows x64, use an x64-to-ARM64 Visual Studio
developer shell and ``cibuildwheel`` with ``CIBW_ARCHS_WINDOWS=ARM64``.
Select ``cp311-win_arm64`` or ``cp312-win_arm64`` via ``CIBW_BUILD``, enabling
``TRITON_STABLE_ABI=ON`` for the latter. Set ``TRITON_APPEND_CMAKE_ARGS`` to
``-DLLVM_NATIVE_TOOL_DIR=<host-LLVM-bin>`` using the x64 tools from the same
pinned LLVM revision. Preserve Windows path separators with an inner quoted
argument, for example:

.. code-block:: powershell

      $Env:TRITON_APPEND_CMAKE_ARGS = "'-DLLVM_NATIVE_TOOL_DIR=C:\path\to\host-llvm\bin'"

``cibuildwheel`` supplies the target Python import
libraries, while the build downloads ARM64 LLVM, AMD codegen, CUDA, and TinyCC
dependencies. Native ARM64 hardware is required to run the resulting wheels.

Windows does not require ``make``: use ``cmake --build`` with the build directory
reported by ``python.build_helpers.get_cmake_dir()`` for incremental native
builds, and ``python -m pytest -s --tb=short`` for individual Python tests.
Running GPU kernels requires a supported NVIDIA GPU/driver or a compatible
ROCm for Windows installation.

For building with a custom LLVM, review the `Building with a custom LLVM <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ section on Github.

You can then test your installation by running the tests:

.. code-block:: bash

      # One-time setup
      make dev-install

      # To run all tests (requires a GPU)
      make test

      # Or, to run tests without a GPU
      make test-nogpu
