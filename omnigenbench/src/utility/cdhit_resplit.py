"""Dataset re-splitting with cd-hit-est clustering.

This utility clusters the combined training/validation/test splits using
``cd-hit-est`` and then reassigns entire clusters to splits while preserving the
original split proportions as closely as possible. The intent is to eliminate
near-duplicate leakage between splits without skewing the dataset balance.

Example command line usage::

    python -m omnigenbench.src.utility.cdhit_resplit \
        --train train.fasta \
        --validation val.fasta \
        --test test.fasta \
        --threshold 0.90 \
        --output-dir results/resplit

The script generates new FASTA files (``train_resplit.fasta`` etc.) containing
the reassigned sequences, along with an optional JSON summary and per-cluster
assignment report.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import subprocess
import tempfile

from .cdhit_similarity import (
    _ensure_executable,
    _extract_identifier_from_clstr,
    _guess_word_size,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceRecord:
    """Representation of a single FASTA entry linked to its original split."""

    split: str
    original_id: str
    description: str
    sequence: str


@dataclass
class Cluster:
    index: int
    members: List[str]  # Prefixed identifiers


def _wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(
        sequence[i : i + width] for i in range(0, len(sequence), width)
    ) or ""


def _parse_fasta_with_prefix(
    fasta_path: Path, split_name: str
) -> Tuple[Dict[str, SequenceRecord], List[str]]:
    """Parse a FASTA file and return prefixed identifiers with metadata."""

    records: Dict[str, SequenceRecord] = {}
    order: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as f:
        current_header: Optional[str] = None
        sequence_lines: List[str] = []
        entry_index = 0

        def flush_entry() -> None:
            nonlocal entry_index
            if current_header is None:
                return
            sequence = "".join(sequence_lines).replace("\n", "").strip()
            if not sequence:
                LOGGER.warning(
                    "Sequence '%s' in split '%s' is empty; skipping.",
                    current_header,
                    split_name,
                )
                return
            entry_index += 1
            header_content = current_header.strip()
            if header_content:
                parts = header_content.split(maxsplit=1)
                original_id = parts[0]
                description = parts[1] if len(parts) > 1 else ""
            else:
                original_id = f"unnamed_{entry_index}"
                description = ""

            prefixed_id = f"{split_name}|{original_id}"
            records[prefixed_id] = SequenceRecord(
                split=split_name,
                original_id=original_id,
                description=description,
                sequence=sequence,
            )
            order.append(prefixed_id)

        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush_entry()
                current_header = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line.strip())

        flush_entry()

    return records, order


def _collect_sequences(
    split_fastas: Dict[str, Path]
) -> Tuple[Dict[str, SequenceRecord], List[str], Dict[str, int]]:
    """Load all provided FASTA files and return shared metadata containers."""

    combined_records: Dict[str, SequenceRecord] = {}
    combined_order: List[str] = []
    split_counts: Dict[str, int] = {}

    for split, fasta_path in split_fastas.items():
        records, order = _parse_fasta_with_prefix(fasta_path, split)
        for prefixed_id, record in records.items():
            if prefixed_id in combined_records:
                LOGGER.warning(
                    "Duplicate identifier '%s' encountered; keeping first instance.",
                    prefixed_id,
                )
                continue
            combined_records[prefixed_id] = record
        combined_order.extend(order)
        split_counts[split] = len(order)

    return combined_records, combined_order, split_counts


def _write_prefixed_fasta(
    combined_order: Sequence[str],
    records: Dict[str, SequenceRecord],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for prefixed_id in combined_order:
            record = records[prefixed_id]
            if record.description:
                f.write(f">{prefixed_id} {record.description}\n")
            else:
                f.write(f">{prefixed_id}\n")
            f.write(_wrap_sequence(record.sequence))
            f.write("\n")


def _run_cd_hit_est(
    input_fasta: Path,
    output_prefix: Path,
    threshold: float,
    threads: int,
    memory_limit: int,
    word_size: Optional[int],
    binary: str,
) -> None:
    _ensure_executable(binary, "cd-hit-est clustering")
    if word_size is None:
        word_size = _guess_word_size(threshold)

    cmd = [
        binary,
        "-i",
        str(input_fasta),
        "-c",
        f"{threshold:.2f}",
        "-n",
        str(word_size),
        "-T",
        str(threads),
        "-M",
        str(memory_limit),
        "-d",
        "0",
        "-o",
        str(output_prefix),
    ]

    LOGGER.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_clusters(clstr_path: Path) -> List[Cluster]:
    clusters: List[Cluster] = []
    current_members: List[str] = []
    cluster_index = -1

    with clstr_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if current_members:
                    clusters.append(Cluster(index=cluster_index, members=current_members))
                cluster_index = int(line.split()[1])
                current_members = []
            else:
                seq_id = _extract_identifier_from_clstr(line)
                current_members.append(seq_id)

    if current_members:
        clusters.append(Cluster(index=cluster_index, members=current_members))

    return clusters


def _assign_clusters(
    clusters: Sequence[Cluster],
    records: Dict[str, SequenceRecord],
    target_counts: Dict[str, int],
) -> Tuple[Dict[str, int], List[Tuple[Cluster, str]]]:
    """Assign each cluster wholesale to a split, respecting target counts."""

    splits = list(target_counts.keys())
    current_counts = {split: 0 for split in splits}
    cluster_assignments: List[Tuple[Cluster, str]] = []

    sorted_clusters = sorted(clusters, key=lambda c: len(c.members), reverse=True)

    for cluster in sorted_clusters:
        cluster_size = len(cluster.members)
        split_counter = Counter(records[member].split for member in cluster.members)
        preferred_split = (
            split_counter.most_common(1)[0][0] if split_counter else splits[0]
        )

        within_capacity = [
            split
            for split in splits
            if current_counts[split] + cluster_size <= target_counts[split]
        ]

        chosen_split: str
        if within_capacity:
            if preferred_split in within_capacity:
                chosen_split = preferred_split
            else:
                chosen_split = min(
                    within_capacity,
                    key=lambda split: target_counts[split]
                    - (current_counts[split] + cluster_size),
                )
        else:
            overfill_scores = {
                split: (current_counts[split] + cluster_size) - target_counts[split]
                for split in splits
            }
            min_overfill = min(overfill_scores.values())
            candidates = [
                split for split, score in overfill_scores.items() if score == min_overfill
            ]
            if preferred_split in candidates:
                chosen_split = preferred_split
            else:
                chosen_split = candidates[0]

        current_counts[chosen_split] += cluster_size
        cluster_assignments.append((cluster, chosen_split))

    return current_counts, cluster_assignments


def _write_split_fastas(
    output_dir: Path,
    cluster_assignments: Iterable[Tuple[Cluster, str]],
    records: Dict[str, SequenceRecord],
) -> Dict[str, Path]:
    """Write reassigned sequences to split-specific FASTA files."""

    split_to_records: Dict[str, List[SequenceRecord]] = {}
    for cluster, assigned_split in cluster_assignments:
        split_to_records.setdefault(assigned_split, [])
        split_to_records[assigned_split].extend(records[member] for member in cluster.members)

    output_paths: Dict[str, Path] = {}
    for split, recs in split_to_records.items():
        output_path = output_dir / f"{split}_resplit.fasta"
        with output_path.open("w", encoding="utf-8") as f:
            for record in recs:
                if record.description:
                    f.write(f">{record.original_id} {record.description}\n")
                else:
                    f.write(f">{record.original_id}\n")
                f.write(_wrap_sequence(record.sequence))
                f.write("\n")
        output_paths[split] = output_path

    return output_paths


def analyze_and_resplit(
    *,
    train_fasta: Path,
    validation_fasta: Optional[Path],
    test_fasta: Optional[Path],
    threshold: float,
    output_dir: Path,
    cd_hit_binary: str,
    threads: int,
    memory_limit: int,
    word_size: Optional[int],
    cluster_report: Optional[Path],
) -> Dict[str, object]:
    """Main orchestration logic for clustering and split reassignment."""

    provided_splits: Dict[str, Path] = {
        name: path
        for name, path in (
            ("train", train_fasta),
            ("validation", validation_fasta),
            ("test", test_fasta),
        )
        if path is not None
    }

    if not provided_splits:
        raise ValueError("At least one dataset split must be provided for resplitting.")

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_records, combined_order, original_counts = _collect_sequences(
        provided_splits
    )

    LOGGER.info("Original split sizes: %s", original_counts)

    with tempfile.TemporaryDirectory(prefix="cdhit_resplit_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        combined_fasta = tmp_dir / "combined_prefixed.fasta"
        _write_prefixed_fasta(combined_order, combined_records, combined_fasta)

        output_prefix = tmp_dir / "combined_clustering"
        _run_cd_hit_est(
            input_fasta=combined_fasta,
            output_prefix=output_prefix,
            threshold=threshold,
            threads=threads,
            memory_limit=memory_limit,
            word_size=word_size,
            binary=cd_hit_binary,
        )

        clstr_path = Path(f"{output_prefix}.clstr")
        clusters = _parse_clusters(clstr_path)
        LOGGER.info("Parsed %d clusters from cd-hit output.", len(clusters))

        final_counts, cluster_assignments = _assign_clusters(
            clusters, combined_records, original_counts
        )

        output_paths = _write_split_fastas(
            output_dir=output_dir,
            cluster_assignments=cluster_assignments,
            records=combined_records,
        )

        if cluster_report:
            with cluster_report.open("w", encoding="utf-8") as f:
                for cluster, assigned_split in cluster_assignments:
                    member_ids = ", ".join(cluster.members)
                    f.write(
                        f"Cluster {cluster.index}\tassigned={assigned_split}\tmembers={member_ids}\n"
                    )

    summary = {
        "threshold": threshold,
        "original_counts": original_counts,
        "final_counts": final_counts,
        "output_fastas": {split: str(path) for split, path in output_paths.items()},
    }

    LOGGER.info("Final split sizes: %s", final_counts)
    return summary


def _parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster combined datasets with cd-hit-est and reassign clusters to "
            "maintain original split ratios."
        )
    )
    parser.add_argument("--train", type=Path, help="Path to the training FASTA file.")
    parser.add_argument("--validation", type=Path, help="Path to the validation FASTA file.")
    parser.add_argument("--test", type=Path, help="Path to the test FASTA file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Sequence identity threshold for cd-hit-est clustering (default: 0.90).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cdhit_resplit"),
        help="Directory to store the resplit FASTA files and optional reports.",
    )
    parser.add_argument(
        "--cd-hit-binary",
        type=str,
        default="cd-hit-est",
        help="Name or path of the cd-hit-est executable.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to allocate to cd-hit-est (default: 4).",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=0,
        help="Maximum memory in MB for cd-hit-est (default: 0 meaning unlimited).",
    )
    parser.add_argument(
        "--word-size",
        type=int,
        help="Override the cd-hit-est word size (-n). If omitted, inferred automatically.",
    )
    parser.add_argument(
        "--cluster-report",
        type=Path,
        help="Optional path to export per-cluster assignment details (TSV).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to save the resplitting summary as JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )

    parsed = parser.parse_args(args=args)
    if not any((parsed.train, parsed.validation, parsed.test)):
        parser.error("At least one of --train, --validation, or --test must be provided.")
    return parsed


def main(args: Optional[Sequence[str]] = None) -> Dict[str, object]:
    parsed = _parse_args(args)
    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    summary = analyze_and_resplit(
        train_fasta=parsed.train,
        validation_fasta=parsed.validation,
        test_fasta=parsed.test,
        threshold=parsed.threshold,
        output_dir=parsed.output_dir,
        cd_hit_binary=parsed.cd_hit_binary,
        threads=parsed.threads,
        memory_limit=parsed.memory_limit,
        word_size=parsed.word_size,
        cluster_report=parsed.cluster_report,
    )

    if parsed.summary_json:
        with parsed.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info("Summary written to %s", parsed.summary_json)

    LOGGER.info("Resplitting completed successfully.")
    return summary


if __name__ == "__main__":
    main()

