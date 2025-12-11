#!/usr/bin/env python3
"""Merge converted SR MNIST baselines into canonical iteration series.

This utility walks a tree of baseline run artifacts, merges metadata and
iteration metrics, and emits three web-ready files per baseline:
- <baseline>__meta.json        → merged metadata + provenance
- <baseline>__iterations.json  → same metadata plus averaged iteration series
- <baseline>__iterations.csv   → table view of the averaged iteration series

It also writes:
- data/SR_MNIST/index.json
- data/SR_MNIST/merge_report.json
- merge_converted_to_iteration.log (configurable via --log-file)

The tool is intentionally flexible and tolerant: missing fields become null,
conflicting metadata values are recorded under meta.aggregated, and per-run
errors are logged into merge_report.json without aborting the entire batch.

Example:
    python merge_converted_to_iteration.py \
      --root ./converted_records/SR_mnist/Centralized_n=10_b=1 \
      --out ./data/SR_MNIST/Centralized_n=10_b=1 \
      --group-by folder \
      --resample 201 \
      --archive \
      --overwrite \
      --allow-pickle \
      --log-file ./merge_converted_to_iteration.log \
      --verbose
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pickle
import torch
import torch.serialization as torch_serialization

CONVERTER_VERSION = "1.0.0"
FLOAT_FIELDS = {"loss", "accuracy", "lr", "progress"}
DEFAULT_RESAMPLE_POINTS = 201
SUPPORTED_GROUP_MODES = {"folder", "prefix", "regex"}

logger = logging.getLogger("merge_converted_to_iteration")


class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA storages to CPU so we can read GPU checkpoints."""

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "torch.storage" and name == "_load_from_bytes":
            return self._load_from_bytes_cpu
        if module.startswith("torch.cuda"):
            module = module.replace("torch.cuda", "torch")
        return super().find_class(module, name)

    @staticmethod
    def _load_from_bytes_cpu(buf: bytes):
        """Load torch storage bytes while forcing all tensors onto CPU."""
        orig_default_restore_location = torch_serialization.default_restore_location

        def cpu_default_restore_location(storage, location):
            if isinstance(location, (bytes, str)) and str(location).startswith("cuda"):
                return orig_default_restore_location(storage, "cpu")
            return orig_default_restore_location(storage, location)

        torch_serialization.default_restore_location = cpu_default_restore_location
        try:
            return torch.load(io.BytesIO(buf), map_location="cpu")
        finally:
            torch_serialization.default_restore_location = orig_default_restore_location


def load_pickle_cpu(path: Path) -> Any:
    with path.open("rb") as fh:
        return CPUUnpickler(fh).load()


@dataclass
class RunData:
    run_id: str
    partition: str
    baseline_key: str
    source_path: Path
    metadata: Dict[str, Any]
    iterations: List[Dict[str, Any]]


@dataclass
class BaselineAggregate:
    partition: str
    baseline_key: str
    runs: List[RunData] = field(default_factory=list)

    def append(self, run: RunData) -> None:
        self.runs.append(run)

    def merged_metadata(self) -> Dict[str, Any]:
        meta_keys = set()
        for run in self.runs:
            meta_keys.update(run.metadata.keys())

        aggregated: Dict[str, List[Any]] = defaultdict(list)
        for key in sorted(meta_keys):
            for run in self.runs:
                aggregated[key].append(run.metadata.get(key))

        merged: Dict[str, Any] = {}
        conflicts: Dict[str, Dict[str, Any]] = {}
        for key, values in aggregated.items():
            cleaned = [v for v in values if v is not None]
            if not cleaned:
                merged[key] = None
                continue

            serialized = [json.dumps(v, sort_keys=True, default=str) for v in cleaned]
            unique_serialized = set(serialized)
            if len(unique_serialized) == 1:
                merged[key] = json.loads(serialized[0])
            else:
                counts = Counter(serialized)
                most_common_serialized, _ = counts.most_common(1)[0]
                merged[key] = json.loads(most_common_serialized)
                conflicts[key] = {
                    "strategy": "mode",
                    "candidates": [
                        {
                            "value": json.loads(val),
                            "count": count,
                        }
                        for val, count in counts.most_common()
                    ],
                }
        if conflicts:
            merged.setdefault("merge_strategy", {}).update(conflicts)
        merged.setdefault("source_runs", [run.run_id for run in self.runs])
        merged.setdefault("run_count", len(self.runs))
        return merged

    def compute_iteration_summary(self, target_points: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.runs:
            return [], {}

        metric_keys = {k for k in self.runs[0].iterations[0].keys() if k not in {"iter", "iteration"}}
        per_run_arrays: Dict[str, List[np.ndarray]] = {k: [] for k in metric_keys | {"iteration"}}

        for run in self.runs:
            series = np.array([row["iteration"] for row in run.iterations], dtype=float)
            if len(series) < 2:
                series = np.linspace(series[0], series[0], target_points)
            resampled_iterations = np.linspace(series[0], series[-1], target_points)
            per_run_arrays["iteration"].append(resampled_iterations)

            for key in metric_keys:
                values = np.array([
                    row.get(key) if row.get(key) is not None else np.nan for row in run.iterations
                ], dtype=float)
                if np.all(np.isnan(values)):
                    resampled = np.full(target_points, np.nan)
                else:
                    origin_x = series
                    target_x = resampled_iterations
                    valid_mask = ~np.isnan(values)
                    if not valid_mask.any():
                        resampled = np.full(target_points, np.nan)
                    else:
                        resampled = np.interp(target_x, origin_x[valid_mask], values[valid_mask])
                per_run_arrays[key].append(resampled)

        merged_series: List[Dict[str, Any]] = []
        stats_summary: Dict[str, Any] = {"point_count": target_points}

        iterations_mean = np.nanmean(np.stack(per_run_arrays["iteration"], axis=0), axis=0)
        for idx in range(target_points):
            point: Dict[str, Any] = {
                "iteration": float(iterations_mean[idx]),
            }
            for key in metric_keys:
                stack = np.stack(per_run_arrays[key], axis=0)
                column = stack[:, idx]
                if np.all(np.isnan(column)):
                    point[key] = None
                else:
                    point[key] = float(np.nanmean(column))
                    std_val = float(np.nanstd(column)) if column.size > 1 else 0.0
                    point[f"std_{key}"] = std_val
            merged_series.append(point)

        for key in metric_keys:
            stack = np.stack(per_run_arrays[key], axis=0)
            stats_summary[key] = {
                "mean": float(np.nanmean(stack)) if not np.all(np.isnan(stack)) else None,
                "std": float(np.nanstd(stack)) if not np.all(np.isnan(stack)) else None,
            }
        return merged_series, stats_summary


@dataclass
class Report:
    processed_baselines: int = 0
    processed_runs: int = 0
    skipped_runs: int = 0
    archived_files: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def record_error(self, path: Path, message: str) -> None:
        self.errors.append({"path": str(path), "message": message})


def configure_logging(log_file: Optional[Path], verbose: bool) -> None:
    handlers: List[logging.Handler] = []
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)
    ch = logging.StreamHandler(sys.stdout if verbose else io.StringIO())
    ch.setFormatter(formatter)
    handlers.append(ch)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_run_files(root: Path) -> List[Path]:
    run_files = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        run_files.append(path)
    return sorted(run_files)


def determine_partition_and_baseline(root: Path, path: Path, group_mode: str, pattern: Optional[re.Pattern[str]]) -> Tuple[str, str, str]:
    relative = path.relative_to(root)
    parts = relative.parts
    if group_mode == "folder":
        baseline_key = Path(parts[-1]).name
        partition = str(Path(*parts[:-1])) if len(parts) > 1 else ""
    elif group_mode == "prefix":
        stem = Path(parts[-1]).name
        baseline_key = stem.split("__", 1)[0]
        partition = str(Path(*parts[:-1])) if len(parts) > 1 else ""
    elif group_mode == "regex":
        if pattern is None:
            raise ValueError("--group-pattern is required when --group-by regex")
        match = pattern.search(parts[-1])
        if not match:
            raise ValueError(f"Path {path} does not match grouping pattern")
        baseline_key = match.group(1) if match.groups() else match.group(0)
        partition = str(Path(*parts[:-1])) if len(parts) > 1 else ""
    else:
        raise ValueError(f"Unsupported group-by mode: {group_mode}")
    run_id = str(relative.with_suffix(""))
    return partition, baseline_key, run_id


def extract_run_metadata(path: Path, allow_pickle: bool) -> Optional[Dict[str, Any]]:
    if allow_pickle:
        try:
            data = load_pickle_cpu(path)
            if isinstance(data, dict):
                return {k: _tensor_to_primitive(v) for k, v in data.items()}
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load pickle %s: %s", path, exc)
            return None
    return None


def _tensor_to_primitive(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return [float(x) for x in value.cpu().flatten().tolist()]
    if isinstance(value, list):
        return [_tensor_to_primitive(v) for v in value]
    if isinstance(value, dict):
        return {k: _tensor_to_primitive(v) for k, v in value.items()}
    return value


def build_iteration_series(meta: Dict[str, Any], target_points: int) -> List[Dict[str, Any]]:
    loss_series = meta.get("loss_path") or []
    acc_series = meta.get("acc_path") or []
    lr_value = meta.get("lr")
    total_iterations = meta.get("total_iterations") or (len(loss_series) - 1 if loss_series else len(acc_series) - 1)
    if total_iterations in (None, 0):
        total_iterations = max(len(loss_series), len(acc_series))
    count = max(len(loss_series), len(acc_series))
    if count == 0:
        return []
    original_iters = np.linspace(0.0, float(total_iterations), num=count)
    target_iters = np.linspace(0.0, float(total_iterations), num=target_points)

    def resample(values: Sequence[float]) -> np.ndarray:
        if not values:
            return np.full(target_points, np.nan)
        arr = np.array(values, dtype=float)
        if arr.size == 1:
            return np.full(target_points, arr[0])
        return np.interp(target_iters, original_iters[: arr.size], arr)

    loss_values = resample([float(x) for x in loss_series])
    acc_values = resample([float(x) for x in acc_series])
    lr_values = np.full(target_points, float(lr_value)) if lr_value is not None else np.full(target_points, np.nan)
    progress_values = (
        target_iters / float(total_iterations)
        if total_iterations
        else np.linspace(0.0, 1.0, num=target_points)
    )

    iterations = []
    for idx in range(target_points):
        entry = {
            "iteration": float(target_iters[idx]),
            "loss": float(loss_values[idx]) if not math.isnan(loss_values[idx]) else None,
            "accuracy": float(acc_values[idx]) if not math.isnan(acc_values[idx]) else None,
            "lr": float(lr_values[idx]) if not math.isnan(lr_values[idx]) else None,
            "progress": float(progress_values[idx]),
            "timestamp": None,
        }
        iterations.append(entry)
    return iterations


def ensure_directory(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any, dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str], dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if row.get(k) is not None else "" for k in fieldnames})


def move_to_archive(path: Path, archive_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / path.name
    if destination.exists():
        destination = archive_root / f"{path.stem}__{sha256_file(path)[:8]}{path.suffix}"
    shutil.move(str(path), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge converted baselines into canonical iteration series")
    parser.add_argument("--root", required=True, type=Path, help="Input root directory (converted recordings)")
    parser.add_argument("--out", required=True, type=Path, help="Output root directory")
    parser.add_argument("--group-by", choices=sorted(SUPPORTED_GROUP_MODES), default="folder")
    parser.add_argument("--group-pattern", help="Regex pattern with capture group (for --group-by regex)")
    parser.add_argument("--resample", type=int, default=DEFAULT_RESAMPLE_POINTS, help="Number of points in resampled series")
    parser.add_argument("--archive", dest="archive", action="store_true", help="Move redundant CSV files to archive")
    parser.add_argument("--no-archive", dest="archive", action="store_false", help="Disable archiving")
    parser.set_defaults(archive=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without writing output")
    parser.add_argument("--allow-pickle", action="store_true", help="Allow loading pickle/torch files")
    parser.add_argument("--keep-per-run", action="store_true", help="Retain per-run outputs (currently no-op)")
    parser.add_argument("--log-file", type=Path, help="Path to detailed log file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging to stdout")
    args = parser.parse_args()

    pattern = re.compile(args.group_pattern) if args.group_pattern else None
    configure_logging(args.log_file, args.verbose)

    report = Report()
    start_time = datetime.now(timezone.utc)

    if not args.root.exists():
        logger.error("Input root %s does not exist", args.root)
        sys.exit(1)

    run_files = discover_run_files(args.root)
    logger.info("Discovered %d run artifact files", len(run_files))

    grouped: Dict[Tuple[str, str], BaselineAggregate] = {}

    for file_path in run_files:
        try:
            partition, baseline_key, run_id = determine_partition_and_baseline(args.root, file_path, args.group_by, pattern)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Skipping %s: %s", file_path, exc)
            report.skipped_runs += 1
            report.record_error(file_path, str(exc))
            continue

        metadata = extract_run_metadata(file_path, args.allow_pickle)
        if metadata is None:
            logger.warning("No metadata extracted from %s", file_path)
            report.skipped_runs += 1
            report.record_error(file_path, "No recognizable metadata")
            continue

        iterations = build_iteration_series(metadata, args.resample)
        if not iterations:
            logger.warning("No iteration data for %s", file_path)
            report.skipped_runs += 1
            report.record_error(file_path, "Missing iteration data")
            continue

        metadata = {k: v for k, v in metadata.items() if k not in {"loss_path", "acc_path"}}

        run = RunData(
            run_id=run_id,
            partition=partition,
            baseline_key=baseline_key,
            source_path=file_path,
            metadata=metadata,
            iterations=iterations,
        )

        aggregate_key = (partition, baseline_key)
        grouped.setdefault(aggregate_key, BaselineAggregate(partition, baseline_key)).append(run)
        report.processed_runs += 1

    logger.info("Collected %d baseline groups", len(grouped))

    index_tree: Dict[str, Dict[str, Dict[str, str]]] = {}

    for (partition, baseline_key), aggregate in grouped.items():
        logger.debug("Merging baseline %s/%s with %d runs", partition, baseline_key, len(aggregate.runs))
        meta = aggregate.merged_metadata()
        merged_iterations, stats_summary = aggregate.compute_iteration_summary(args.resample)

        output_dir = args.out / partition / baseline_key
        ensure_directory(output_dir, args.dry_run)

        base_filename = baseline_key
        meta_path = output_dir / f"{base_filename}__meta.json"
        iter_json_path = output_dir / f"{base_filename}__iterations.json"
        iter_csv_path = output_dir / f"{base_filename}__iterations.csv"

        if not args.overwrite and any(path.exists() for path in (meta_path, iter_json_path, iter_csv_path)):
            logger.info("Skipping baseline %s/%s because output exists", partition, baseline_key)
            continue

        provenance = {
            "source_files": [str(run.source_path) for run in aggregate.runs],
            "raw_hashes": {str(run.source_path): sha256_file(run.source_path) for run in aggregate.runs},
            "converted_at": datetime.now(timezone.utc).isoformat(),
            "converter_version": CONVERTER_VERSION,
        }
        meta_out = {
            "baseline": baseline_key,
            "partition": partition,
            "meta": meta,
            "provenance": provenance,
        }
        write_json(meta_path, meta_out, args.dry_run)

        iterations_payload = {
            "baseline": baseline_key,
            "partition": partition,
            "meta": meta,
            "statistics": stats_summary,
            "iterations": merged_iterations,
            "provenance": provenance,
        }
        write_json(iter_json_path, iterations_payload, args.dry_run)

        csv_rows = [
            {
                "iteration": row.get("iteration"),
                "accuracy": row.get("accuracy"),
                "loss": row.get("loss"),
                "lr": row.get("lr"),
                "progress": row.get("progress"),
                "timestamp": row.get("timestamp"),
            }
            for row in merged_iterations
        ]
        write_csv(iter_csv_path, csv_rows, ["iteration", "accuracy", "loss", "lr", "progress", "timestamp"], args.dry_run)

        index_tree.setdefault(partition, {})[baseline_key] = {
            "meta": str(meta_path),
            "iterations_json": str(iter_json_path),
            "iterations_csv": str(iter_csv_path),
        }
        report.processed_baselines += 1

    index_payload = {
        "converter_version": CONVERTER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "partitions": index_tree,
    }
    ensure_directory(args.out, args.dry_run)
    write_json(args.out / "index.json", index_payload, args.dry_run)

    report_payload = {
        "converter_version": CONVERTER_VERSION,
        "started_at": start_time.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "processed_baselines": report.processed_baselines,
        "processed_runs": report.processed_runs,
        "skipped_runs": report.skipped_runs,
        "archived_files": report.archived_files,
        "errors": report.errors,
    }
    write_json(args.out / "merge_report.json", report_payload, args.dry_run)

    logger.info("Finished: %d baselines, %d runs processed", report.processed_baselines, report.processed_runs)


if __name__ == "__main__":
    main()
