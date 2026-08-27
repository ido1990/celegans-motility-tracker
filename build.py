"""Build a portable single-file executable via PyInstaller.

NOTE: PyInstaller builds are OS-specific. Running this on Linux produces a Linux
binary, not a Windows .exe. To distribute a .exe to non-technical Windows users,
run this same script on a Windows machine (with `pip install -r requirements.txt`
done there first).
"""
import subprocess
import sys

# scipy's optional array-api-compat shim has torch/cupy/jax backends that PyInstaller's
# scipy hook speculatively bundles if those packages happen to be installed on the build
# machine, even though this app never imports them. Exclude explicitly to keep the build fast.
_UNRELATED_HEAVY_MODULES = ["torch", "torchvision", "sklearn", "nvidia", "triton", "jax", "cupy"]


def main():
    args = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--noconsole",
        "--name", "MotilityTracker",
    ]
    for mod in _UNRELATED_HEAVY_MODULES:
        args += ["--exclude-module", mod]
    args.append("gui.py")
    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
