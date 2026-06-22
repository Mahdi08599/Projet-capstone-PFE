"""
Quick setup verification — run this after installing dependencies
to check that everything works.

Usage:
    python check_setup.py
"""

import sys

print("=" * 50)
print("SETUP VERIFICATION")
print("=" * 50)

errors = []

# Python version
v = sys.version_info
print(f"\nPython : {v.major}.{v.minor}.{v.micro}", end="")
if v.minor >= 10:
    print(" ✓")
else:
    print(" ✗ (need >= 3.10)")
    errors.append("Python >= 3.10 required")

# Core libraries
libs = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "torch": "PyTorch",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pyarrow": "pyarrow",
    "tqdm": "tqdm",
}

for module, name in libs.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "?")
        print(f"{name:15s} : {version} ✓")
    except ImportError:
        print(f"{name:15s} : NOT INSTALLED ✗")
        errors.append(f"{name} not installed")

# GPU check
try:
    import torch
    if torch.cuda.is_available():
        print(f"\nGPU : {torch.cuda.get_device_name(0)} ✓")
        print(f"      CUDA {torch.version.cuda}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\nGPU : Apple MPS ✓")
    else:
        print(f"\nGPU : Not available (CPU only)")
        print("      The model will train slower but it will work.")
except Exception:
    pass

# Data check
import os
raw_dir = os.path.join("data", "raw")
expected_files = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]

print(f"\nData files in {raw_dir}/ :")
for f in expected_files:
    path = os.path.join(raw_dir, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {f:30s} : {size_mb:.0f} Mo ✓")
    else:
        print(f"  {f:30s} : MISSING ✗")
        errors.append(f"Missing: {path}")

# Summary
print("\n" + "=" * 50)
if errors:
    print(f"ISSUES FOUND ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL CHECKS PASSED ✓")
    print("You're ready to run: python src/data_preprocessing.py")
print("=" * 50)
