# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Hardware-agnostic kernel dispatch layer for edge AI acceleration.

Provides a lightweight runtime dispatch system that lets optimized hardware kernels be registered, discovered, and
adopted incrementally without fragmenting the core inference codebase. The inference pipeline asks the dispatcher for
the best available implementation of an operation on the current hardware; the dispatcher handles hardware detection,
capability matching, priority selection, and PyTorch fallback when no optimized kernel matches.

This module is self-contained: it depends only on the Python stdlib and PyTorch. The LOGGER, get_cpu_info, and
is_intel helpers are implemented inline so the file can be copied into any project without an ultralytics install.

Examples:
    Register and resolve an optimized CUDA NMS kernel
    >>> from kernel_dispatch import dispatcher, register_kernel, resolve_kernel
    >>> dispatcher.clear()
    >>> def cuda_nms(boxes, scores, iou_threshold):
    ...     return boxes  # placeholder
    >>> _ = register_kernel("nms", "cuda", cuda_nms, min_compute_capability=(8, 6), priority=10)
    >>> fn = resolve_kernel("nms")  # returns cuda_nms on Ada/Ampere CUDA, None elsewhere
    >>> dispatcher.clear()
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch

LOGGER = logging.getLogger("kernel_dispatch")


@lru_cache(maxsize=1)
def get_cpu_info() -> str:
    """Return a human-readable CPU name string, e.g. 'Apple M4 Pro' or 'Intel Core i7-9750H 2.60GHz'.

    Queries platform-specific sources (sysctl on macOS, /proc/cpuinfo on Linux, registry on Windows) with
    generic fallbacks so a string is always returned.
    """
    try:
        if platform.system() == "Darwin":
            s = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            if s:
                return _clean_cpu_name(s)
        elif platform.system() == "Linux":
            p = Path("/proc/cpuinfo")
            if p.exists():
                for line in p.read_text(errors="ignore").splitlines():
                    if "model name" in line:
                        return _clean_cpu_name(line.split(":", 1)[1])
        elif platform.system() == "Windows":
            try:
                import winreg as wr

                with wr.OpenKey(wr.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                    val, _ = wr.QueryValueEx(k, "ProcessorNameString")
                    if val:
                        return _clean_cpu_name(val)
            except Exception:
                pass
        s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
        return _clean_cpu_name(s or "Unknown CPU")
    except Exception:
        s = platform.processor() or platform.machine() or ""
        return _clean_cpu_name(s or "Unknown CPU")


def _clean_cpu_name(s: str) -> str:
    """Normalize a raw CPU descriptor string into a clean human-readable name."""
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
    if m := re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.I):
        return f"{m.group(1)} {m.group(2)}"
    if m := re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.I):
        return f"{m.group(1)} {m.group(2)}"
    return s


def is_intel() -> bool:
    """Return True if the system has an Intel CPU (or Intel GPU via xpu-smi)."""
    if "intel" in get_cpu_info().lower():
        return True
    try:
        result = subprocess.run(["xpu-smi", "discovery"], capture_output=True, text=True, timeout=5)
        return "intel" in result.stdout.lower()
    except Exception:
        return False

# CUDA compute capability -> NVIDIA architecture codename. Covers consumer RTX families targeted by the NorthStar
# reference kernel plus common datacenter/Jetson parts; unknown capabilities fall back to None so unmatched kernels
# are simply skipped rather than misrouted.
_CUDA_ARCH_BY_CC: dict[tuple[int, int], str] = {
    (5, 0): "Maxwell",
    (5, 2): "Maxwell",
    (5, 3): "Maxwell",  # Jetson Nano
    (6, 0): "Pascal",
    (6, 1): "Pascal",  # GTX 10 series
    (6, 2): "Pascal",
    (7, 0): "Volta",  # V100
    (7, 2): "Volta",  # Jetson Xavier
    (7, 5): "Turing",  # RTX 20 series, T4
    (8, 0): "Ampere",  # A100
    (8, 6): "Ampere",  # RTX 30 series, A40, A6000
    (8, 7): "Ampere",  # Jetson Orin
    (8, 9): "Ada Lovelace",  # RTX 40 series, L4, L40
    (9, 0): "Hopper",  # H100
    (10, 0): "Blackwell",
}


@dataclass
class DeviceCapabilities:
    """Describe the runtime capabilities of a compute device for kernel selection.

    Attributes:
        type (str): Torch device type, e.g. 'cuda', 'mps', 'cpu', 'npu', 'xpu'.
        vendor (str): Hardware vendor, e.g. 'nvidia', 'apple', 'intel', 'amd', 'qualcomm', 'unknown'.
        name (str): Human-readable device name (e.g. 'NVIDIA GeForce RTX 4090', 'Apple M2', 'Intel Core i7').
        compute_capability (tuple[int, int] | None): CUDA (major, minor) compute capability, or None off CUDA.
        arch (str | None): Architecture codename (e.g. 'Ada Lovelace') when derivable, else None.
        supports_fp16 (bool): Whether FP16 inference is supported on this device.
        supports_int8 (bool): Whether INT8 inference is supported on this device.
        supports_bf16 (bool): Whether BF16 inference is supported on this device.
    """

    type: str
    vendor: str
    name: str
    compute_capability: tuple[int, int] | None = None
    arch: str | None = None
    supports_fp16: bool = False
    supports_int8: bool = False
    supports_bf16: bool = False

    def matches(self, backend: str | None, hardware: str | None, min_cc: tuple[int, int] | None) -> bool:
        """Return True if this device satisfies a kernel's backend/hardware/compute-capability constraints."""
        if backend and self.type != backend:
            return False
        if hardware and self.arch != hardware and hardware not in (self.name,):
            return False
        if min_cc and (self.compute_capability is None or self.compute_capability < min_cc):
            return False
        return True


def _cuda_capabilities(index: int) -> DeviceCapabilities:
    """Build DeviceCapabilities for a CUDA device from torch.cuda.get_device_properties."""
    props = torch.cuda.get_device_properties(index)
    cc = (props.major, props.minor)
    return DeviceCapabilities(
        type="cuda",
        vendor="nvidia",
        name=props.name,
        compute_capability=cc,
        arch=_CUDA_ARCH_BY_CC.get(cc),
        supports_fp16=True,
        supports_int8=cc >= (7, 0),  # INT8 tensor cores arrive with Volta/Turing+; older GPUs lack efficient INT8
        supports_bf16=cc >= (8, 0),  # BF16 native support starts at Ampere
    )


def detect_device(device: str | int | torch.device | None = None) -> DeviceCapabilities:
    """Detect capabilities of the current (or requested) compute device.

    Uses inline get_cpu_info / is_intel helpers and torch's own device probes. Unknown devices return a minimal
    CPU-flavored capability so they remain fully functional via the PyTorch fallback path.

    Args:
        device (str | int | torch.device | None): Device to inspect. None/''/'cpu' inspects the host CPU; an int or
            'cuda:N' inspects that CUDA device; 'mps' inspects Apple Metal.

    Returns:
        (DeviceCapabilities): Capabilities of the resolved device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(str(device))

    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if index < torch.cuda.device_count():
            return _cuda_capabilities(index)
        # Out-of-range CUDA request: fall through to CPU rather than crash; callers resolve via select_device upstream
        LOGGER.warning(f"kernel_dispatch: CUDA index {index} unavailable, falling back to CPU capabilities")

    if device.type == "mps":
        return DeviceCapabilities(
            type="mps",
            vendor="apple",
            name=get_cpu_info(),  # Apple Silicon: CPU name reflects the SoC (e.g. 'Apple M2')
            supports_fp16=True,
            supports_int8=True,
            supports_bf16=False,
        )

    if device.type == "npu":
        return DeviceCapabilities(
            type="npu", vendor="huawei", name="Ascend NPU", supports_fp16=True, supports_int8=True
        )

    cpu_name = get_cpu_info()
    vendor = (
        "intel"
        if is_intel()
        else "amd"
        if "amd" in cpu_name.lower()
        else "apple"
        if "apple" in cpu_name.lower()
        else "unknown"
    )
    return DeviceCapabilities(
        type=device.type if device.type in {"cpu", "xpu", "tpu", "vulkan"} else "cpu",
        vendor=vendor,
        name=cpu_name,
        supports_fp16=True,  # CPU FP16 is emulated but always available via PyTorch fallback
        supports_int8=True,
        supports_bf16=False,
    )


@dataclass
class KernelEntry:
    """A registered optimized kernel implementation and its hardware constraints.

    Attributes:
        operation (str): Operation name, e.g. 'nms', 'sigmoid', 'postprocess'.
        backend (str): Required device type, e.g. 'cuda', 'mps', 'cpu'.
        implementation (Callable): The kernel function called on match.
        priority (int): Higher priority wins among matching kernels; ties break by registration order (last wins).
        min_compute_capability (tuple[int, int] | None): Minimum CUDA (major, minor) required, or None.
        precision (str | None): Required device precision support, one of 'fp16', 'int8', 'bf16', or None.
        hardware (str | None): Required architecture codename or exact device name, or None for any.
    """

    operation: str
    backend: str
    implementation: Callable[..., Any]
    priority: int = 0
    min_compute_capability: tuple[int, int] | None = None
    precision: str | None = None
    hardware: str | None = None

    def matches(self, caps: DeviceCapabilities) -> bool:
        """Return True if the given device capabilities satisfy this kernel's constraints."""
        if not caps.matches(self.backend, self.hardware, self.min_compute_capability):
            return False
        if self.precision == "fp16" and not caps.supports_fp16:
            return False
        if self.precision == "int8" and not caps.supports_int8:
            return False
        if self.precision == "bf16" and not caps.supports_bf16:
            return False
        return True


class KernelRegistry:
    """Thread-safe registry of optimized kernels keyed by operation name.

    Methods:
        register: Add a KernelEntry; later registrations of the same operation/backend override earlier ones.
        resolve: Return the highest-priority matching implementation for an operation on a device, or None for fallback.
        entries: Return a snapshot list of all registered KernelEntry objects.
        clear: Remove all registered kernels (intended for tests).
    """

    def __init__(self) -> None:
        """Initialize an empty registry with a reentrant lock for concurrent registration from multiple threads."""
        self._entries: list[KernelEntry] = []
        self._lock = threading.RLock()

    def register(self, entry: KernelEntry) -> None:
        """Register a kernel entry, overriding any prior entry with the same operation and backend."""
        with self._lock:
            self._entries = [
                e for e in self._entries if not (e.operation == entry.operation and e.backend == entry.backend)
            ]
            self._entries.append(entry)

    def resolve(self, operation: str, caps: DeviceCapabilities | None = None) -> Callable[..., Any] | None:
        """Resolve the best matching kernel implementation for an operation on a device.

        Args:
            operation (str): Operation name to resolve.
            caps (DeviceCapabilities | None): Device capabilities to match against. None auto-detects the current device.

        Returns:
            (Callable | None): The matching kernel implementation, or None to signal the caller should use the
            built-in PyTorch fallback path.
        """
        if caps is None:
            caps = detect_device()
        with self._lock:
            # Stable sort by priority ascending then reverse so the last-registered highest-priority entry wins ties,
            # letting later registrations override earlier ones without an explicit priority bump.
            candidates = sorted(
                (e for e in self._entries if e.operation == operation and e.matches(caps)),
                key=lambda e: e.priority,
            )
        return candidates[-1].implementation if candidates else None

    def entries(self) -> list[KernelEntry]:
        """Return a shallow copy of all registered entries for introspection and tests."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        """Remove all registered kernels; intended for test isolation."""
        with self._lock:
            self._entries.clear()


# Module-level singleton dispatcher, mirroring the package's callback-registry pattern.
dispatcher = KernelRegistry()


def register_kernel(
    operation: str,
    backend: str,
    implementation: Callable[..., Any],
    priority: int = 0,
    min_compute_capability: tuple[int, int] | None = None,
    precision: str | None = None,
    hardware: str | None = None,
) -> KernelEntry:
    """Register an optimized kernel with the global dispatcher.

    Args:
        operation (str): Operation name, e.g. 'nms'.
        backend (str): Required device type, e.g. 'cuda'.
        implementation (Callable): Kernel function invoked on match.
        priority (int): Higher wins among matches; ties break by registration order (last wins).
        min_compute_capability (tuple[int, int] | None): Minimum CUDA (major, minor), or None.
        precision (str | None): Required precision support: 'fp16', 'int8', 'bf16', or None.
        hardware (str | None): Required architecture codename or exact device name, or None for any.

    Returns:
        (KernelEntry): The registered entry.
    """
    entry = KernelEntry(
        operation=operation,
        backend=backend,
        implementation=implementation,
        priority=priority,
        min_compute_capability=min_compute_capability,
        precision=precision,
        hardware=hardware,
    )
    dispatcher.register(entry)
    LOGGER.debug(
        f"kernel_dispatch: registered '{operation}' for backend '{backend}' "
        f"(priority={priority}, min_cc={min_compute_capability}, precision={precision}, hardware={hardware})"
    )
    return entry


def resolve_kernel(operation: str, caps: DeviceCapabilities | None = None) -> Callable[..., Any] | None:
    """Resolve the best matching kernel for an operation, or None to use the PyTorch fallback.

    Args:
        operation (str): Operation name to resolve.
        caps (DeviceCapabilities | None): Device capabilities to match against. None auto-detects the current device.

    Returns:
        (Callable | None): Matching kernel implementation, or None when no optimized kernel applies.
    """
    return dispatcher.resolve(operation, caps)
