"""GPU backend — transparent CuPy/NumPy switch.

If CuPy (CUDA) is available, all array operations run on GPU.
Otherwise falls back to NumPy with zero code changes.

Usage:
    from simulation.gpu_backend import xp, GPU_AVAILABLE, gpu_info
    a = xp.zeros((50, 50))  # Works on both GPU and CPU
"""
from __future__ import annotations
import os
import numpy as np

GPU_AVAILABLE = False
xp = np  # Default to numpy, override if cupy works
_backend_name = "numpy (CPU)"


def _try_init_cupy():
    global xp, GPU_AVAILABLE, _backend_name

    # Allow disabling GPU via env var
    if os.environ.get("DISABLE_GPU", "").lower() in ("1", "true", "yes"):
        xp = np
        _backend_name = "numpy (disabled)"
        return

    try:
        import cupy
        # Test basic operation
        test = cupy.zeros(4)
        _ = test.get()  # Force sync
        del test
        xp = cupy
        GPU_AVAILABLE = True
        _backend_name = "cupy (CUDA)"
        print(f"[GPU] CuPy initialized: {cupy.cuda.Device().attributes['DeviceName']}")
    except Exception as e:
        xp = np
        GPU_AVAILABLE = False
        _backend_name = "numpy (CPU)"
        print(f"[GPU] CuPy not available ({type(e).__name__}), using NumPy CPU")


def gpu_info() -> dict:
    """Return GPU status info."""
    info = {"backend": _backend_name, "gpu_available": GPU_AVAILABLE}
    if GPU_AVAILABLE:
        try:
            import cupy
            dev = cupy.cuda.Device()
            info["device_name"] = str(dev.attributes["DeviceName"])
            mem = dev.mem_info
            info["vram_total_gb"] = round(mem[1] / 1e9, 2)
            info["vram_free_gb"] = round(mem[0] / 1e9, 2)
        except Exception:
            pass
    return info


def to_numpy(arr):
    """Convert array to numpy (for serialization/broadcast)."""
    if GPU_AVAILABLE:
        try:
            import cupy
            if isinstance(arr, cupy.ndarray):
                return cupy.asnumpy(arr)
        except Exception:
            pass
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr)


def to_gpu(arr):
    """Move array to GPU if available."""
    if GPU_AVAILABLE:
        try:
            import cupy
            return cupy.asarray(arr)
        except Exception:
            pass
    return arr


# Initialize on import
_try_init_cupy()
