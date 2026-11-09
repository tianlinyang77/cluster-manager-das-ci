"""Build an HCU-enabled NVIDIA Resiliency Extension wheel.

The upstream project is kept as an unmodified git submodule.  A temporary
build tree is assembled for every build, containing:

* the current checkout of ``nvidia_resiliency_ext``;
* the HCU adapter package; and
* a small bootstrap in the upstream package's ``__init__.py``.

This removes the old manual step of editing the submodule in place while
preserving the upstream project's own Poetry build configuration.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / "nvidia_resiliency_ext"
HCU_PACKAGE_DIR = ROOT / "hcu_resiliency_ext"

_PACKAGES_BLOCK = '''packages = [
    { include = "nvidia_resiliency_ext", from = "src" },
    { include = "hcu_resiliency_ext", from = "src" },
]'''

_TELEMETRY_DEPENDENCIES = '''nv-one-logger-core = { version = "*", optional = true }
nv-one-logger-training-telemetry = { version = "*", optional = true }
'''

_TELEMETRY_EXTRA = '''telemetry = [
    "nv-one-logger-core",
    "nv-one-logger-training-telemetry",
]
'''

_BOOTSTRAP = '''

# HCU_RESILIENCY_EXT_AUTO_BOOTSTRAP
# This block is injected into the temporary build tree only.  The git
# submodule itself is never modified.
try:
    from hcu_resiliency_ext.adaptor import nre_adaptor as _hcu_nre_adaptor  # noqa: F401
except ImportError:
    # Keep the upstream package usable when the adapter is not installed.
    pass
'''


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "build",
            "dist",
        ),
    )


def _upstream_version() -> str:
    """Return a PEP 440 version derived from the submodule commit."""

    try:
        describe = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={UPSTREAM_DIR}",
                "-C",
                str(UPSTREAM_DIR),
                "describe",
                "--tags",
                "--long",
                "--always",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0.0.0+local"

    match = re.fullmatch(r"v?(\d+\.\d+\.\d+)(?:-main)?-(\d+)-g([0-9a-f]+)", describe)
    if match:
        base, distance, commit = match.groups()
        return base if distance == "0" else f"{base}.dev{distance}+g{commit}"

    release = re.fullmatch(r"v?(\d+\.\d+\.\d+)(?:-main)?", describe)
    if release:
        return release.group(1)
    return "0.0.0+local"


def prepare_build_tree(build_root: Path) -> Path:
    """Create and return an isolated upstream build tree.

    ``build_root`` must be a fresh directory.  The returned directory is
    suitable for passing to ``pip wheel`` and contains no references to the
    source checkout at runtime.
    """

    upstream_src = UPSTREAM_DIR / "src"
    upstream_pyproject = UPSTREAM_DIR / "pyproject.toml"
    if not upstream_src.is_dir() or not upstream_pyproject.is_file():
        raise FileNotFoundError(
            "nvidia_resiliency_ext submodule is not initialized; run "
            "git submodule update --init --remote nvidia_resiliency_ext"
        )
    if not HCU_PACKAGE_DIR.is_dir():
        raise FileNotFoundError(f"HCU adapter package not found: {HCU_PACKAGE_DIR}")

    staged = build_root / "nvidia_resiliency_ext"
    _copy_tree(UPSTREAM_DIR, staged)

    staged_src = staged / "src"
    _copy_tree(HCU_PACKAGE_DIR, staged_src / "hcu_resiliency_ext")

    pyproject_path = staged / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    upstream_packages = '''packages = [
    { include = "nvidia_resiliency_ext", from = "src" },
]'''
    if upstream_packages not in pyproject:
        raise RuntimeError(
            "upstream pyproject.toml changed: package declaration was not found"
        )
    pyproject = pyproject.replace(upstream_packages, _PACKAGES_BLOCK, 1)
    # The temporary tree intentionally has no .git directory.  Disable the
    # upstream VCS version plugin and carry the source commit in the wheel
    # version instead of requiring a manual edit or a fake .git directory.
    pyproject = pyproject.replace(
        'version = "0.0.0"  # Replaced at build time by poetry-dynamic-versioning',
        f'version = "{_upstream_version()}"',
        1,
    )
    pyproject = pyproject.replace(
        "[tool.poetry-dynamic-versioning]\nenable = true",
        "[tool.poetry-dynamic-versioning]\nenable = false",
        1,
    )
    # Keep One Logger optional in the generated wheel.  The source submodule
    # remains untouched; this metadata is injected only into the staging tree.
    if "nv-one-logger-core" not in pyproject:
        pyproject = pyproject.replace(
            "[tool.poetry.dependencies]\n",
            "[tool.poetry.dependencies]\n" + _TELEMETRY_DEPENDENCIES,
            1,
        )
        pyproject = pyproject.replace(
            "[tool.poetry.extras]\n",
            "[tool.poetry.extras]\n" + _TELEMETRY_EXTRA,
            1,
        )
    pyproject_path.write_text(pyproject, encoding="utf-8")

    # The upstream build helper passes Windows backslash paths to protoc as
    # paths relative to an unspecified working directory.  Normalize those
    # paths in the temporary tree so the same build command works on Windows
    # and POSIX hosts.
    build_script = staged / "build.py"
    build_text = build_script.read_text(encoding="utf-8")
    build_text = build_text.replace(
        'proto_dir = os.path.join("src", "nvidia_resiliency_ext", "shared_utils", "proto")',
        'proto_dir = os.path.join(os.path.dirname(__file__), "src", "nvidia_resiliency_ext", "shared_utils", "proto")',
        1,
    )
    old_proto_paths = '''proto_paths_rel_to_src = [
        os.path.join("nvidia_resiliency_ext", "shared_utils", "proto", f) for f in existing_protos
    ]'''
    new_proto_paths = '''proto_paths = [os.path.join(proto_dir, f) for f in existing_protos]'''
    if old_proto_paths not in build_text:
        raise RuntimeError("upstream build.py changed: protoc path declaration was not found")
    build_text = build_text.replace(old_proto_paths, new_proto_paths, 1)
    build_text = build_text.replace("*proto_paths_rel_to_src,", "*proto_paths,", 1)
    build_script.write_text(build_text, encoding="utf-8")

    init_path = staged_src / "nvidia_resiliency_ext" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    if "HCU_RESILIENCY_EXT_AUTO_BOOTSTRAP" not in init_text:
        init_path.write_text(_BOOTSTRAP.lstrip() + "\n" + init_text, encoding="utf-8")

    return staged


def build_wheel(output_dir: Path) -> Path:
    """Build the combined wheel and return its output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep build staging files under the caller's current project directory
    # instead of the system temporary directory.  This makes the build easy
    # to inspect and works in environments where /tmp is not persistent.
    temp_root = Path.cwd() / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nvrx-hcu-build-", dir=str(temp_root)
    ) as temp_dir:
        staged = prepare_build_tree(Path(temp_dir))
        env = os.environ.copy()
        # HCU deployments generally do not provide CUDA/CUPTI.  Callers can
        # explicitly set this to 0/on/off as needed for the upstream extension.
        env.setdefault("STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD", "1")
        # The upstream build helper prints a checkmark; force UTF-8 so the
        # build is also reliable under Windows locales such as GBK.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        # Also redirect pip's isolated-build and ephemeral-wheel files.  pip
        # honors these variables on both Windows and POSIX systems.
        temp_root_value = str(temp_root.resolve())
        env["TMP"] = temp_root_value
        env["TEMP"] = temp_root_value
        env["TMPDIR"] = temp_root_value
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--disable-pip-version-check",
                "--wheel-dir",
                str(output_dir.resolve()),
                str(staged),
            ],
            check=True,
            cwd=staged,
            env=env,
        )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build nvidia-resiliency-ext with the HCU adapter injected automatically."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "dist-nvidia",
        help="directory receiving the wheel (default: dist-nvidia)",
    )
    args = parser.parse_args(argv)
    output_dir = build_wheel(args.outdir)
    print(f"Built HCU-enabled nvidia-resiliency-ext wheel in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
