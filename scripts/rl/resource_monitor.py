#!/usr/bin/env python3
"""Independent 1 Hz process/GPU monitor for P1 benchmarks."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2


class NvmlError(RuntimeError):
    pass


def nvml_device_count() -> int:
    """Return the number of physical NVML devices without creating a CUDA context."""

    library = ctypes.CDLL("libnvidia-ml.so.1")
    code = library.nvmlInit_v2()
    if code != NVML_SUCCESS:
        raise NvmlError(f"nvmlInit_v2 failed with NVML status {code}")
    try:
        count = ctypes.c_uint()
        code = library.nvmlDeviceGetCount_v2(ctypes.byref(count))
        if code != NVML_SUCCESS:
            raise NvmlError(f"nvmlDeviceGetCount_v2 failed with NVML status {code}")
        return int(count.value)
    finally:
        library.nvmlShutdown()


class _MemoryInfoV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _ProcessInfoV3(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class NvmlDevice:
    """Small ctypes NVML wrapper; avoids invoking `nvidia-smi` every second."""

    def __init__(self, index: int):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._bind()
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )

    def _bind(self) -> None:
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemoryInfoV2)]
        self.lib.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Utilization)]
        self.lib.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        self.lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != NVML_SUCCESS:
            raise NvmlError(f"{operation} failed with NVML status {code}")

    def close(self) -> None:
        self._check(self.lib.nvmlShutdown(), "nvmlShutdown")

    def _processes(self, symbol: str) -> list[dict[str, int]]:
        function = getattr(self.lib, symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(_ProcessInfoV3),
        ]
        count = ctypes.c_uint(0)
        code = function(self.handle, ctypes.byref(count), None)
        if code == NVML_SUCCESS:
            return []
        if code != NVML_ERROR_INSUFFICIENT_SIZE:
            self._check(code, symbol)
        values = (_ProcessInfoV3 * max(1, count.value))()
        self._check(function(self.handle, ctypes.byref(count), values), symbol)
        return [
            {"pid": int(item.pid), "used_gpu_memory_bytes": int(item.used_gpu_memory)}
            for item in values[: count.value]
        ]

    def sample(self) -> dict[str, object]:
        memory = _MemoryInfoV2()
        memory.version = ctypes.sizeof(_MemoryInfoV2) | (2 << 24)
        utilization = _Utilization()
        temperature = ctypes.c_uint()
        power_mw = ctypes.c_uint()
        sm_clock = ctypes.c_uint()
        memory_clock = ctypes.c_uint()
        uuid = ctypes.create_string_buffer(96)
        name = ctypes.create_string_buffer(96)
        self._check(self.lib.nvmlDeviceGetMemoryInfo_v2(self.handle, ctypes.byref(memory)), "memory")
        self._check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)), "utilization")
        self._check(
            self.lib.nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU, ctypes.byref(temperature)),
            "temperature",
        )
        self._check(self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power_mw)), "power")
        self._check(self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM, ctypes.byref(sm_clock)), "sm_clock")
        self._check(
            self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_MEM, ctypes.byref(memory_clock)),
            "memory_clock",
        )
        self._check(self.lib.nvmlDeviceGetUUID(self.handle, uuid, len(uuid)), "uuid")
        self._check(self.lib.nvmlDeviceGetName(self.handle, name, len(name)), "name")
        return {
            "gpu_uuid": uuid.value.decode("ascii"),
            "gpu_name": name.value.decode("utf-8"),
            "gpu_utilization_percent": int(utilization.gpu),
            "memory_utilization_percent": int(utilization.memory),
            "memory_total_bytes": int(memory.total),
            "memory_reserved_bytes": int(memory.reserved),
            "memory_used_bytes": int(memory.used),
            "temperature_c": int(temperature.value),
            "power_w": power_mw.value / 1000.0,
            "sm_clock_mhz": int(sm_clock.value),
            "memory_clock_mhz": int(memory_clock.value),
            "compute_processes": self._processes("nvmlDeviceGetComputeRunningProcesses_v3"),
            "graphics_processes": self._processes("nvmlDeviceGetGraphicsRunningProcesses_v3"),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--contention-file", type=Path, required=True)
    parser.add_argument("--allowed-other-gpu-pid", type=int, action="append", default=[])
    parser.add_argument("--period-s", type=float, default=1.0)
    return parser.parse_args()


def _target_processes(target_pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(target_pid)
        values = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []
    monitor_pid = os.getpid()
    return [process for process in values if process.pid != monitor_pid]


def _process_sample(target_pid: int) -> tuple[dict[str, object], set[int]]:
    processes = _target_processes(target_pid)
    rss = vms = 0
    cpu_percent = 0.0
    alive_pids: set[int] = set()
    for process in processes:
        try:
            memory = process.memory_info()
            rss += memory.rss
            vms += memory.vms
            cpu_percent += process.cpu_percent(interval=None)
            alive_pids.add(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    virtual_memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return (
        {
            "target_alive": bool(processes),
            "process_count": len(alive_pids),
            "process_pids": json.dumps(sorted(alive_pids), separators=(",", ":")),
            "rss_bytes": rss,
            "vms_bytes": vms,
            "cpu_percent": cpu_percent,
            "system_available_bytes": virtual_memory.available,
            "swap_used_bytes": swap.used,
        },
        alive_pids,
    )


def _unexpected_gpu_pids(gpu_pids: set[int], target_pids: set[int], allowed_other_pids: set[int]) -> set[int]:
    return gpu_pids - target_pids - allowed_other_pids


def _gpu_memory_by_pid(*process_groups: list[dict[str, int]]) -> dict[int, int]:
    """Deduplicate NVML compute/graphics entries and retain each PID's largest value."""

    memory_by_pid: dict[int, int] = {}
    for processes in process_groups:
        for process in processes:
            pid = int(process["pid"])
            memory_by_pid[pid] = max(memory_by_pid.get(pid, 0), int(process["used_gpu_memory_bytes"]))
    return memory_by_pid


def main() -> int:
    args = _parse_args()
    if args.period_s <= 0.0:
        raise ValueError("period-s must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_other_pids = set(args.allowed_other_gpu_pid)
    nvml = NvmlDevice(args.gpu)
    process_fields = [
        "monotonic_s",
        "utc",
        "target_alive",
        "process_count",
        "process_pids",
        "rss_bytes",
        "vms_bytes",
        "cpu_percent",
        "system_available_bytes",
        "swap_used_bytes",
    ]
    gpu_fields = [
        "monotonic_s",
        "utc",
        "gpu_uuid",
        "gpu_name",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "memory_total_bytes",
        "memory_reserved_bytes",
        "memory_used_bytes",
        "temperature_c",
        "power_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "compute_processes",
        "graphics_processes",
        "target_gpu_processes",
        "target_gpu_pids",
        "target_gpu_memory_bytes",
        "other_gpu_processes",
        "other_gpu_pids",
        "other_gpu_memory_bytes",
        "unattributed_gpu_memory_bytes",
    ]
    try:
        with (args.output_dir / "process_samples.csv").open("w", newline="", encoding="utf-8") as process_stream, (
            args.output_dir / "gpu_samples.csv"
        ).open("w", newline="", encoding="utf-8") as gpu_stream:
            process_writer = csv.DictWriter(process_stream, fieldnames=process_fields)
            gpu_writer = csv.DictWriter(gpu_stream, fieldnames=gpu_fields)
            process_writer.writeheader()
            gpu_writer.writeheader()
            next_sample = time.monotonic()
            while True:
                timestamp = time.monotonic()
                utc = datetime.now(timezone.utc).isoformat()
                process_row, target_pids = _process_sample(args.target_pid)
                gpu_row = nvml.sample()
                gpu_pids = {
                    item["pid"]
                    for key in ("compute_processes", "graphics_processes")
                    for item in gpu_row[key]
                }
                memory_by_pid = _gpu_memory_by_pid(gpu_row["compute_processes"], gpu_row["graphics_processes"])
                target_gpu_processes = [
                    {"pid": pid, "used_gpu_memory_bytes": memory_by_pid[pid]}
                    for pid in sorted(gpu_pids & target_pids)
                ]
                other_gpu_processes = [
                    {"pid": pid, "used_gpu_memory_bytes": memory_by_pid[pid]}
                    for pid in sorted(gpu_pids - target_pids)
                ]
                unexpected_gpu_pids = _unexpected_gpu_pids(gpu_pids, target_pids, allowed_other_pids)
                if unexpected_gpu_pids and not args.contention_file.exists():
                    args.contention_file.write_text(
                        json.dumps(
                            {
                                "detected_monotonic_s": timestamp,
                                "detected_utc": utc,
                                "unexpected_gpu_pids": sorted(unexpected_gpu_pids),
                                "allowed_other_gpu_pids": sorted(allowed_other_pids),
                                "target_pids": sorted(target_pids),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                gpu_row["compute_processes"] = json.dumps(gpu_row["compute_processes"], separators=(",", ":"))
                gpu_row["graphics_processes"] = json.dumps(gpu_row["graphics_processes"], separators=(",", ":"))
                gpu_row["target_gpu_processes"] = json.dumps(target_gpu_processes, separators=(",", ":"))
                gpu_row["target_gpu_pids"] = json.dumps(
                    [process["pid"] for process in target_gpu_processes], separators=(",", ":")
                )
                gpu_row["target_gpu_memory_bytes"] = sum(
                    process["used_gpu_memory_bytes"] for process in target_gpu_processes
                )
                gpu_row["other_gpu_processes"] = json.dumps(other_gpu_processes, separators=(",", ":"))
                gpu_row["other_gpu_pids"] = json.dumps(sorted(gpu_pids - target_pids), separators=(",", ":"))
                gpu_row["other_gpu_memory_bytes"] = sum(
                    process["used_gpu_memory_bytes"] for process in other_gpu_processes
                )
                gpu_row["unattributed_gpu_memory_bytes"] = max(
                    0, int(gpu_row["memory_used_bytes"]) - sum(memory_by_pid.values())
                )
                process_writer.writerow({"monotonic_s": timestamp, "utc": utc, **process_row})
                gpu_writer.writerow({"monotonic_s": timestamp, "utc": utc, **gpu_row})
                process_stream.flush()
                gpu_stream.flush()
                if args.stop_file.exists() or not process_row["target_alive"]:
                    break
                next_sample += args.period_s
                time.sleep(max(0.0, next_sample - time.monotonic()))
    finally:
        nvml.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
