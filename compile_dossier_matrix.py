#!/usr/bin/env python3
"""
Forensic compilation gate for the In-Band Transformation Experiment.

Reads raw_telemetry_dump.log, segments each lift block by boundary signatures,
extracts hardware-derived metrics via rigid regular expressions, and emits an
un-compromised dossier matrix. Missing metrics are flagged as DATA_MISSING.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent
TELEMETRY_LOG = WORKSPACE_ROOT / "raw_telemetry_dump.log"
DOSSIER_OUTPUT = WORKSPACE_ROOT / "dossier_matrix.txt"

DATA_MISSING = "DATA_MISSING"

LIFT_START_PATTERN = re.compile(r"^=== LIFT (\d+) START ===$")
LIFT_END_PATTERN = re.compile(r"^=== LIFT (\d+) CONCLUDED ===$")

METRIC_PATTERNS: Dict[str, re.Pattern[str]] = {
    "lift": re.compile(r"^\[METRIC_LOG\] Lift=(\d+)$"),
    "payload": re.compile(r"^\[METRIC_LOG\] Payload=(\S+)$"),
    "lane": re.compile(r"^\[METRIC_LOG\] Lane=(\S+)$"),
    "ttft": re.compile(r"^\[METRIC_LOG\] Captured_TTFT=(.+)$"),
    "itl": re.compile(r"^\[METRIC_LOG\] Captured_ITL=(.+)$"),
    "itl_array": re.compile(r"^\[METRIC_LOG\] ITL_Array=(.*)$"),
    "pols": re.compile(r"^\[METRIC_LOG\] PoLS=(.+)$"),
    "ingestion_entropy_ratio": re.compile(
        r"^\[METRIC_LOG\] Ingestion_Entropy_Ratio=(.+)$"
    ),
    "activation_density": re.compile(
        r"^\[METRIC_LOG\] Activation_Density=(.+)$"
    ),
    "footprint_reduction": re.compile(
        r"^\[METRIC_LOG\] Context_Footprint_Reduction=(.+)$"
    ),
    "sequence_entropy": re.compile(r"^\[METRIC_LOG\] Sequence_Entropy=(.+)$"),
    "qc": re.compile(r"^\[METRIC_LOG\] QC=(.+)$"),
    "reverse_audit_fault": re.compile(
        r"^\[METRIC_LOG\] Reverse_Audit_Fault=(.+)$"
    ),
}

COMPACT_METRIC_PATTERN = re.compile(
    r"^\[METRIC_LOG\] Lift=(\d+)\s+Payload=(\S+)\s+Lane=(\S+)\s+"
    r"Captured_TTFT=(\S+)\s+Captured_ITL=(\S+)\s+"
    r"Activation_Density=(\S+)\s+PoLS=(\S+)\s+"
    r"Ingestion_Entropy_Ratio=(\S+)\s+Context_Footprint_Reduction=(\S+)$"
)

REQUIRED_METRIC_KEYS = (
    "ttft",
    "itl",
    "activation_density",
    "pols",
    "ingestion_entropy_ratio",
    "footprint_reduction",
)


@dataclass
class LiftRecord:
    lift_id: int
    payload: str = DATA_MISSING
    lane: str = DATA_MISSING
    ttft: str = DATA_MISSING
    itl: str = DATA_MISSING
    itl_array: str = DATA_MISSING
    pols: str = DATA_MISSING
    ingestion_entropy_ratio: str = DATA_MISSING
    activation_density: str = DATA_MISSING
    footprint_reduction: str = DATA_MISSING
    sequence_entropy: str = DATA_MISSING
    qc: str = DATA_MISSING
    reverse_audit_fault: str = DATA_MISSING
    block_found: bool = False
    parse_errors: List[str] = field(default_factory=list)

    def validate_integrity(self) -> None:
        for key in REQUIRED_METRIC_KEYS:
            value = getattr(self, key)
            if value == DATA_MISSING or value.strip() == "":
                setattr(self, key, DATA_MISSING)
                self.parse_errors.append(f"Missing required metric: {key}")

    @property
    def is_valid(self) -> bool:
        return all(getattr(self, key) != DATA_MISSING for key in REQUIRED_METRIC_KEYS)


def initialize_lift_registry() -> Dict[int, LiftRecord]:
    return {lift_id: LiftRecord(lift_id=lift_id) for lift_id in range(1, 13)}


def parse_compact_metric_line(line: str, record: LiftRecord) -> None:
    match = COMPACT_METRIC_PATTERN.match(line.strip())
    if not match:
        return
    (
        lift_str,
        payload,
        lane,
        ttft,
        itl,
        activation_density,
        pols,
        ingestion_entropy_ratio,
        footprint_reduction,
    ) = match.groups()
    record.lift_id = int(lift_str)
    record.payload = payload
    record.lane = lane
    record.ttft = ttft
    record.itl = itl
    record.activation_density = activation_density
    record.pols = pols
    record.ingestion_entropy_ratio = ingestion_entropy_ratio
    record.footprint_reduction = footprint_reduction


def parse_expanded_metric_line(line: str, record: LiftRecord) -> None:
    stripped = line.strip()
    for field_name, pattern in METRIC_PATTERNS.items():
        match = pattern.match(stripped)
        if match:
            setattr(record, field_name, match.group(1))
            return


def segment_log_blocks(log_text: str) -> Dict[int, List[str]]:
    blocks: Dict[int, List[str]] = {}
    current_lift: Optional[int] = None
    current_lines: List[str] = []

    for raw_line in log_text.splitlines():
        line = raw_line.rstrip("\n")
        start_match = LIFT_START_PATTERN.match(line.strip())
        end_match = LIFT_END_PATTERN.match(line.strip())

        if start_match:
            if current_lift is not None and current_lines:
                blocks[current_lift] = current_lines
            current_lift = int(start_match.group(1))
            current_lines = [line]
            continue

        if current_lift is not None:
            current_lines.append(line)
            if end_match:
                end_lift = int(end_match.group(1))
                if end_lift == current_lift:
                    blocks[current_lift] = current_lines
                    current_lift = None
                    current_lines = []

    if current_lift is not None and current_lines:
        blocks[current_lift] = current_lines

    return blocks


def parse_lift_block(lift_id: int, lines: List[str]) -> LiftRecord:
    record = LiftRecord(lift_id=lift_id, block_found=True)

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("[METRIC_LOG]"):
            continue
        parse_compact_metric_line(stripped, record)
        parse_expanded_metric_line(stripped, record)

    record.validate_integrity()
    return record


def format_dossier_row(record: LiftRecord) -> str:
    def fmt(value: str) -> str:
        if value == DATA_MISSING:
            return f"**{DATA_MISSING}**"
        return value

    return (
        f"| {record.lift_id:>2} "
        f"| {fmt(record.payload):<18} "
        f"| {fmt(record.lane):<18} "
        f"| {fmt(record.ttft):>14} "
        f"| {fmt(record.itl):>14} "
        f"| {fmt(record.activation_density):>18} "
        f"| {fmt(record.pols):>14} "
        f"| {fmt(record.ingestion_entropy_ratio):>22} "
        f"| {fmt(record.footprint_reduction):>24} |"
    )


def build_dossier_header() -> str:
    header = (
        "| Lift | Payload            | Lane               | "
        "Captured_TTFT  | Captured_ITL   | Activation_Density | "
        "PoLS           | Ingestion_Entropy_Ratio | "
        "Context_Footprint_Reduction |"
    )
    separator = (
        "|-----:|:-------------------|:-------------------|"
        "---------------:|---------------:|-------------------:|"
        "---------------:|------------------------:|"
        "-------------------------:|"
    )
    return f"{header}\n{separator}"


def compile_dossier(log_path: Path) -> str:
    registry = initialize_lift_registry()

    if not log_path.exists():
        for lift_id in registry:
            registry[lift_id].parse_errors.append("Telemetry log file not found")
        lines = [
            "# DOSSIER MATRIX — FORENSIC COMPILATION",
            f"# Source: {log_path} [NOT FOUND]",
            "",
            build_dossier_header(),
        ]
        for lift_id in range(1, 13):
            lines.append(format_dossier_row(registry[lift_id]))
        lines.append("")
        lines.append(f"Valid lifts: 0/12")
        return "\n".join(lines)

    log_text = log_path.read_text(encoding="utf-8")
    blocks = segment_log_blocks(log_text)

    for lift_id in range(1, 13):
        if lift_id in blocks:
            registry[lift_id] = parse_lift_block(lift_id, blocks[lift_id])
        else:
            registry[lift_id].parse_errors.append(
                "No boundary block found in telemetry log"
            )
            registry[lift_id].validate_integrity()

    valid_count = sum(1 for r in registry.values() if r.is_valid)

    output_lines = [
        "# DOSSIER MATRIX — FORENSIC COMPILATION",
        f"# Source: {log_path}",
        f"# Lifts segmented: {len(blocks)}/12",
        f"# Valid lifts: {valid_count}/12",
        "",
        build_dossier_header(),
    ]

    for lift_id in range(1, 13):
        output_lines.append(format_dossier_row(registry[lift_id]))

    output_lines.append("")
    output_lines.append("## Integrity Summary")
    output_lines.append("")

    for lift_id in range(1, 13):
        record = registry[lift_id]
        if record.is_valid:
            status = "VALID"
        else:
            status = f"**{DATA_MISSING}**"
        errors = "; ".join(record.parse_errors) if record.parse_errors else "none"
        output_lines.append(
            f"- Lift {lift_id:>2}: {status} — block_found={record.block_found} — {errors}"
        )

    output_lines.append("")
    return "\n".join(output_lines)


def main() -> int:
    print("[COMPILER] Forensic dossier compilation starting...", flush=True)

    dossier_text = compile_dossier(TELEMETRY_LOG)
    DOSSIER_OUTPUT.write_text(dossier_text, encoding="utf-8")

    print(dossier_text, flush=True)
    print(f"[COMPILER] Dossier written to {DOSSIER_OUTPUT}", flush=True)

    missing_count = dossier_text.count(f"**{DATA_MISSING}**")
    if missing_count > 0:
        print(
            f"[COMPILER] Integrity gate: {missing_count} DATA_MISSING field(s) detected.",
            flush=True,
        )
        return 1

    print("[COMPILER] Integrity gate: all 12 lifts fully populated.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
