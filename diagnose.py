"""Hardware diagnostics for Neuro-Snake."""
import platform
import sys

print("=" * 60)
print("NEURO-SNAKE HARDWARE DIAGNOSTIC")
print("=" * 60)
print(f"Python:        {sys.version.split()[0]}")
print(f"Platform:      {platform.platform()}")
print(f"Processor:     {platform.processor()}")
print()

import torch
print(f"PyTorch:       {torch.__version__}")
print(f"PyTorch build: {'+cuda' if '+cu' in torch.__version__ else '+cpu or other'}")
print()

# ─── CUDA ────────────────────────────────────────────────────────────────
print("CUDA (NVIDIA GPU)")
print(f"  available:       {torch.cuda.is_available()}")
print(f"  cuda version:    {torch.version.cuda}")
print(f"  device count:    {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  device name:     {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  memory:          {props.total_memory / 1024**3:.2f} GB")
    print(f"  compute cap:     {props.major}.{props.minor}")
else:
    print("  → no CUDA device visible to PyTorch")
    print("    fix: install CUDA-enabled torch, e.g.")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
print()

# ─── MPS ─────────────────────────────────────────────────────────────────
print("MPS (Apple Silicon)")
mps = getattr(torch.backends, "mps", None)
print(f"  module present:  {mps is not None}")
print(f"  available:       {bool(mps and mps.is_available())}")
print()

# ─── XPU ─────────────────────────────────────────────────────────────────
print("XPU (Intel Arc / Core Ultra)")
xpu_present = hasattr(torch, "xpu")
print(f"  module present:  {xpu_present}")
if xpu_present:
    try:
        avail = torch.xpu.is_available()
        print(f"  available:       {avail}")
        if avail:
            print(f"  device count:    {torch.xpu.device_count()}")
            print(f"  device name:     {torch.xpu.get_device_name(0)}")
    except Exception as exc:
        print(f"  error: {exc}")
else:
    print("  → torch was not built with XPU support")
    print("    fix: pip install torch --index-url https://download.pytorch.org/whl/xpu")
print()

# ─── NPU (Intel NPU on Core Ultra 100H/200V) ─────────────────────────────
print("NPU (Intel Core Ultra / Ascend)")
npu_avail = False
try:
    if hasattr(torch, "npu") and torch.npu.is_available():
        npu_avail = True
        print(f"  Ascend NPU:      available")
        print(f"  device name:     {torch.npu.get_device_name(0)}")
except Exception as exc:
    print(f"  Ascend NPU:      {exc}")

try:
    import intel_npu_acceleration_library as inal
    print(f"  Intel NPU lib:   installed (version {getattr(inal, '__version__', '?')})")
    npu_avail = True
except ImportError:
    print("  Intel NPU lib:   not installed")
    print("    fix: pip install intel-npu-acceleration-library")

if not npu_avail:
    print("  → no NPU backend reachable")
print()

# ─── DirectML ────────────────────────────────────────────────────────────
print("DirectML (Windows, any GPU)")
try:
    import torch_directml
    print(f"  installed:       yes")
    print(f"  available:       {torch_directml.is_available()}")
    if torch_directml.is_available():
        print(f"  device count:    {torch_directml.device_count()}")
        print(f"  device name:     {torch_directml.device_name(0)}")
except ImportError:
    print("  installed:       no")
    print("    fix: pip install torch-directml")
print()

# ─── What the project will use ───────────────────────────────────────────
import config
selected = config.detect_device()
print("=" * 60)
print(f"PROJECT WILL USE: {config.device_name(selected)}")
print("=" * 60)