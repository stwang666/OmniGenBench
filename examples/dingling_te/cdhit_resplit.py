"""Dataset re-splitting with cd-hit-est clustering.

This utility clusters the combined training/validation/test splits using
``cd-hit-est`` and then reassigns entire clusters to splits while preserving the
original split proportions as closely as possible. The intent is to eliminate
near-duplicate leakage between splits without skewing the dataset balance.

Example command line usage::

        python cdhit_resplit_1.py \
      --train train.csv \
      --validation valid.csv \
      --test test.csv \
      --sequence-column seq \
      --id-column ID \
      --threshold 0.80 \
      --output-dir cdhit_resplit_data_1

The script generates new CSV files (``train_resplit.csv`` etc.) containing
the reassigned sequences, along with an optional JSON summary and per-cluster
assignment report.

To reuse clustering results with different split ratios, use the companion script
``cdhit_reassign.py`` with the saved cluster files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import subprocess
import tempfile

from cdhit_similarity import (
    _ensure_executable,
    _extract_identifier_from_clstr,
    _guess_word_size,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceRecord:
    """Representation of a single CSV row linked to its original split."""

    split: str
    original_id: str
    sequence: str
    row_data: Dict[str, str]  # All columns from the original CSV


@dataclass
class Cluster:
    index: int
    members: List[str]  # Prefixed identifiers


def _wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(
        sequence[i : i + width] for i in range(0, len(sequence), width)
    ) or ""


def _parse_csv_with_prefix(
    csv_path: Path, split_name: str, sequence_column: str, id_column: Optional[str]
) -> Tuple[Dict[str, SequenceRecord], List[str], List[str], List[Tuple[int, str]]]:
    """Parse a CSV file and return prefixed identifiers with metadata.
    
    Returns:
        records: Dictionary mapping prefixed IDs to SequenceRecord objects
        order: List of prefixed IDs in order
        headers: List of column headers
        skipped_rows: List of (row_index, original_id) tuples for rows with empty sequences
    """

    records: Dict[str, SequenceRecord] = {}
    order: List[str] = []
    headers: List[str] = []
    skipped_rows: List[Tuple[int, str]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} appears to be empty or invalid.")
        
        headers = list(reader.fieldnames)
        
        if sequence_column not in headers:
            raise ValueError(
                f"Sequence column '{sequence_column}' not found in {csv_path}. "
                f"Available columns: {headers}"
            )
        
        entry_index = 0
        for row in reader:
            entry_index += 1
            sequence = row[sequence_column].strip()
            
            # Determine the original ID first
            if id_column and id_column in headers and row[id_column]:
                original_id = str(row[id_column]).strip()
            else:
                original_id = f"row_{entry_index}"
            
            if not sequence:
                skipped_rows.append((entry_index, original_id))
                LOGGER.warning(
                    "Row %d (ID: %s) in split '%s' has empty sequence; skipping.",
                    entry_index,
                    original_id,
                    split_name,
                )
                continue
            
            prefixed_id = f"{split_name}|{original_id}"
            
            records[prefixed_id] = SequenceRecord(
                split=split_name,
                original_id=original_id,
                sequence=sequence,
                row_data=dict(row),
            )
            order.append(prefixed_id)

    return records, order, headers, skipped_rows


def _collect_sequences(
    split_csvs: Dict[str, Path],
    sequence_column: str,
    id_column: Optional[str],
) -> Tuple[Dict[str, SequenceRecord], List[str], Dict[str, int], Dict[str, int], List[str], Dict[str, List[Tuple[int, str]]], Dict[str, List[Tuple[str, str]]]]:
    """Load all provided CSV files and return shared metadata containers.
    
    Returns:
        combined_records: All valid sequence records (deduplicated)
        combined_order: Order of all valid records (deduplicated)
        original_counts_before_dedup: Count of valid sequences per split (before deduplication)
        deduplicated_counts: Count of valid sequences per split (after deduplication)
        all_headers: CSV column headers
        skipped_by_split: Dictionary mapping split names to lists of skipped (row_index, id) tuples
        duplicates_by_split: Dictionary mapping split names to lists of duplicate (original_id, sequence) tuples
    """

    combined_records: Dict[str, SequenceRecord] = {}
    combined_order: List[str] = []
    original_counts_before_dedup: Dict[str, int] = {}
    deduplicated_counts: Dict[str, int] = {}
    all_headers: List[str] = []
    skipped_by_split: Dict[str, List[Tuple[int, str]]] = {}
    duplicates_by_split: Dict[str, List[Tuple[str, str]]] = {}
    
    # Track seen (id, sequence) pairs for deduplication
    seen_id_sequence_pairs: Dict[Tuple[str, str], str] = {}  # (id, sequence) -> first_prefixed_id

    for split, csv_path in split_csvs.items():
        records, order, headers, skipped_rows = _parse_csv_with_prefix(
            csv_path, split, sequence_column, id_column
        )
        
        # Store headers from the first file
        if not all_headers:
            all_headers = headers
        
        # Count before deduplication (valid sequences only, excluding skipped)
        original_counts_before_dedup[split] = len(order)
        
        duplicates_by_split[split] = []
        
        for prefixed_id in order:
            record = records[prefixed_id]
            id_seq_pair = (record.original_id, record.sequence)
            
            # Check if this (id, sequence) pair has been seen before
            if id_seq_pair in seen_id_sequence_pairs:
                first_prefixed_id = seen_id_sequence_pairs[id_seq_pair]
                duplicates_by_split[split].append((record.original_id, record.sequence))
                LOGGER.warning(
                    "Duplicate (ID, sequence) pair found: ID='%s', sequence='%s...' "
                    "in split '%s' (prefixed_id='%s'). Already seen in '%s'. Skipping duplicate.",
                    record.original_id,
                    record.sequence[:50],
                    split,
                    prefixed_id,
                    first_prefixed_id,
                )
                continue
            
            # Check for duplicate prefixed_id (shouldn't happen with proper prefixing, but just in case)
            if prefixed_id in combined_records:
                LOGGER.warning(
                    "Duplicate prefixed identifier '%s' encountered; keeping first instance.",
                    prefixed_id,
                )
                continue
            
            # Add to combined records
            combined_records[prefixed_id] = record
            combined_order.append(prefixed_id)
            seen_id_sequence_pairs[id_seq_pair] = prefixed_id
        
        # Count after deduplication
        deduplicated_counts[split] = sum(1 for pid in combined_order if combined_records[pid].split == split)
        skipped_by_split[split] = skipped_rows

    return combined_records, combined_order, original_counts_before_dedup, deduplicated_counts, all_headers, skipped_by_split, duplicates_by_split


def _write_prefixed_fasta(
    combined_order: Sequence[str],
    records: Dict[str, SequenceRecord],
    output_path: Path,
) -> None:
    """Write sequences to FASTA format for cd-hit-est processing."""
    with output_path.open("w", encoding="utf-8") as f:
        for prefixed_id in combined_order:
            record = records[prefixed_id]
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
) -> Path:
    """Run cd-hit-est and return the path to the .clstr file."""
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
    
    return Path(f"{output_prefix}.clstr")


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


def _save_clusters(clusters: List[Cluster], output_path: Path) -> None:
    """Save parsed clusters to a pickle file for later reuse."""
    with output_path.open("wb") as f:
        pickle.dump(clusters, f)
    LOGGER.info("Saved %d clusters to %s", len(clusters), output_path)


def _save_clusters_json(clusters: List[Cluster], output_path: Path) -> None:
    """Save parsed clusters to a JSON file for human readability."""
    cluster_data = [
        {"index": c.index, "members": c.members, "size": len(c.members)}
        for c in clusters
    ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=2)
    LOGGER.info("Saved %d clusters (JSON format) to %s", len(clusters), output_path)


def load_clusters(cluster_file: Path) -> List[Cluster]:
    """Load clusters from a pickle file."""
    with cluster_file.open("rb") as f:
        clusters = pickle.load(f)
    LOGGER.info("Loaded %d clusters from %s", len(clusters), cluster_file)
    return clusters


def _assign_clusters(
    clusters: Sequence[Cluster],
    records: Dict[str, SequenceRecord],
    target_counts: Dict[str, int],
) -> Tuple[Dict[str, int], List[Tuple[Cluster, str]]]:
    """将整簇分配到各 split，尽量贴合原始（去重前）的目标计数。

    策略：
    - 使用配额(remaining quota)驱动：remaining = target - current
    - 选择使 |remaining - cluster_size| 最小的 split（即最小化分配后的剩余额度绝对值）
    - 若存在多候选，优先 remaining 较大的，再优先该簇的原始多数 split
    - 仍出现并列时使用稳定顺序（splits 列表顺序）
    这样能更稳定地逼近目标计数，避免按比例贪心导致的极端偏移。
    """

    splits = list(target_counts.keys())
    current_counts = {split: 0 for split in splits}
    cluster_assignments: List[Tuple[Cluster, str]] = []

    # 由大到小处理簇，先处理大簇更有利于贴近配额
    sorted_clusters = sorted(clusters, key=lambda c: len(c.members), reverse=True)

    for cluster in sorted_clusters:
        cluster_size = len(cluster.members)

        # 该簇内原始 split 的多数派，用作平票优先
        split_counter = Counter(records[member].split for member in cluster.members)
        preferred_split = split_counter.most_common(1)[0][0] if split_counter else splits[0]

        # 计算每个 split 的剩余额度以及分配此簇后的代价
        best_split = None
        best_key = None  # (overshoot_first, abs_remaining_after, -remaining_before, preferred_flag)

        for split in splits:
            remaining_before = target_counts[split] - current_counts[split]
            remaining_after = remaining_before - cluster_size

            # overshoot 表示是否会超配；优先避免超配
            overshoot_first = 1 if remaining_after < 0 else 0

            # 主要目标：最小化 |remaining_after|
            abs_remaining_after = abs(remaining_after)

            # 次目标：优先消耗剩余额度多的 split（-remaining_before 越小越好）
            negate_remaining_before = -remaining_before

            preferred_flag = 0 if split == preferred_split else 1

            key = (overshoot_first, abs_remaining_after, negate_remaining_before, preferred_flag)

            if best_key is None or key < best_key:
                best_key = key
                best_split = split

        chosen_split = best_split if best_split is not None else preferred_split
        current_counts[chosen_split] += cluster_size
        cluster_assignments.append((cluster, chosen_split))

    return current_counts, cluster_assignments


def _write_split_csvs(
    output_dir: Path,
    cluster_assignments: Iterable[Tuple[Cluster, str]],
    records: Dict[str, SequenceRecord],
    headers: List[str],
) -> Dict[str, Path]:
    """Write reassigned sequences to split-specific CSV files."""

    split_to_records: Dict[str, List[SequenceRecord]] = {}
    for cluster, assigned_split in cluster_assignments:
        split_to_records.setdefault(assigned_split, [])
        split_to_records[assigned_split].extend(records[member] for member in cluster.members)

    output_paths: Dict[str, Path] = {}
    for split, recs in split_to_records.items():
        output_path = output_dir / f"{split}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for record in recs:
                writer.writerow(record.row_data)
        output_paths[split] = output_path

    return output_paths


def _write_detailed_report(
    report_path: Path,
    cluster_assignments: List[Tuple[Cluster, str]],
    records: Dict[str, SequenceRecord],
    skipped_by_split: Dict[str, List[Tuple[int, str]]],
    duplicates_by_split: Dict[str, List[Tuple[str, str]]],
    original_counts_before_dedup: Dict[str, int],
    deduplicated_counts: Dict[str, int],
    final_counts: Dict[str, int],
) -> None:
    """Write a detailed resplitting report including cluster statistics, skipped samples, and duplicates."""
    
    # Organize clusters by assigned split
    split_to_clusters: Dict[str, List[int]] = {}
    for cluster, assigned_split in cluster_assignments:
        split_to_clusters.setdefault(assigned_split, [])
        split_to_clusters[assigned_split].append(cluster.index)
    
    # Calculate proportions
    total_before_dedup = sum(original_counts_before_dedup.values())
    total_deduplicated = sum(deduplicated_counts.values())
    total_final = sum(final_counts.values())
    
    proportions_before_dedup = {split: count / total_before_dedup for split, count in original_counts_before_dedup.items()}
    proportions_deduplicated = {split: count / total_deduplicated for split, count in deduplicated_counts.items()}
    final_proportions = {split: count / total_final for split, count in final_counts.items()}
    
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET RE-SPLITTING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("CLUSTERING SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total clusters: {len(cluster_assignments)}\n")
        f.write(f"Total sequences clustered: {sum(len(c.members) for c, _ in cluster_assignments)}\n\n")
        
        # Split proportion comparison
        f.write("SPLIT PROPORTION COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Split':<15} {'Original':<15} {'%':<10} {'After Dedup':<15} {'%':<10} {'Final Count':<15} {'Final %':<10} {'vs Original':<12} {'vs Dedup':<12}\n")
        f.write("-" * 80 + "\n")
        for split in sorted(original_counts_before_dedup.keys()):
            before_count = original_counts_before_dedup[split]
            before_pct = proportions_before_dedup[split] * 100
            dedup_count = deduplicated_counts.get(split, 0)
            dedup_pct = proportions_deduplicated.get(split, 0) * 100 if deduplicated_counts.get(split, 0) > 0 else 0
            final_count = final_counts.get(split, 0)
            final_pct = final_proportions.get(split, 0) * 100
            deviation_from_original = final_pct - before_pct
            deviation_from_dedup = final_pct - dedup_pct if dedup_count > 0 else 0
            f.write(f"{split:<15} {before_count:<15} {before_pct:<10.2f} {dedup_count:<15} {dedup_pct:<10.2f} {final_count:<15} {final_pct:<10.2f} {deviation_from_original:+.2f}% {deviation_from_dedup:+.2f}%\n")
        f.write("\n")
        
        # Cluster distribution by split
        f.write("CLUSTER DISTRIBUTION BY SPLIT\n")
        f.write("-" * 80 + "\n")
        for split in sorted(split_to_clusters.keys()):
            cluster_indices = sorted(split_to_clusters[split])
            num_clusters = len(cluster_indices)
            num_sequences = sum(
                len(cluster.members) 
                for cluster, assigned_split in cluster_assignments 
                if assigned_split == split
            )
            f.write(f"\n{split.upper()}:\n")
            f.write(f"  Number of clusters: {num_clusters}\n")
            f.write(f"  Number of sequences: {num_sequences}\n")
            f.write(f"  Cluster indices: {cluster_indices[:20]}")
            if len(cluster_indices) > 20:
                f.write(f" ... (and {len(cluster_indices) - 20} more)")
            f.write("\n")
        
        # Duplicate samples summary (counts only, no detailed list)
        f.write("\n" + "=" * 80 + "\n")
        f.write("DUPLICATE SAMPLES (SAME ID AND SEQUENCE)\n")
        f.write("=" * 80 + "\n")
        total_duplicates = sum(len(dups) for dups in duplicates_by_split.values())
        f.write(f"Total duplicate samples removed: {total_duplicates}\n\n")
        
        for split in sorted(duplicates_by_split.keys()):
            duplicate_rows = duplicates_by_split[split]
            if duplicate_rows:
                f.write(f"{split.upper()}: {len(duplicate_rows)} duplicate samples removed\n")
        
        # Skipped samples summary (counts only, no detailed list)
        f.write("\n" + "=" * 80 + "\n")
        f.write("SKIPPED SAMPLES (EMPTY SEQUENCES)\n")
        f.write("=" * 80 + "\n")
        total_skipped = sum(len(skipped) for skipped in skipped_by_split.values())
        f.write(f"Total skipped samples: {total_skipped}\n\n")
        
        for split in sorted(skipped_by_split.keys()):
            skipped_rows = skipped_by_split[split]
            if skipped_rows:
                f.write(f"{split.upper()}: {len(skipped_rows)} skipped samples\n")
        
        # Detailed cluster assignments
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED CLUSTER ASSIGNMENTS\n")
        f.write("=" * 80 + "\n\n")
        
        for cluster, assigned_split in cluster_assignments:
            # Count original split distribution
            original_split_counts = Counter(records[member].split for member in cluster.members)
            
            f.write(f"Cluster {cluster.index}:\n")
            f.write(f"  Assigned to: {assigned_split}\n")
            f.write(f"  Size: {len(cluster.members)}\n")
            f.write(f"  Original split distribution: {dict(original_split_counts)}\n")
            f.write(f"  Sample members (first 5): {cluster.members[:5]}\n")
            if len(cluster.members) > 5:
                f.write(f"  ... and {len(cluster.members) - 5} more\n")
            f.write("\n")


def analyze_and_resplit(
    *,
    train_csv: Path,
    validation_csv: Optional[Path],
    test_csv: Optional[Path],
    sequence_column: str,
    id_column: Optional[str],
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
            ("train", train_csv),
            ("validation", validation_csv),
            ("test", test_csv),
        )
        if path is not None
    }

    if not provided_splits:
        raise ValueError("At least one dataset split must be provided for resplitting.")

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_records, combined_order, original_counts_before_dedup, deduplicated_counts, headers, skipped_by_split, duplicates_by_split = _collect_sequences(
        provided_splits, sequence_column, id_column
    )

    LOGGER.info("Original split sizes (before deduplication): %s", original_counts_before_dedup)
    LOGGER.info("Split sizes after deduplication: %s", deduplicated_counts)
    
    # Calculate and log proportions
    total_before_dedup = sum(original_counts_before_dedup.values())
    total_deduplicated = sum(deduplicated_counts.values())
    
    proportions_before_dedup = {split: count / total_before_dedup for split, count in original_counts_before_dedup.items()}
    proportions_deduplicated = {split: count / total_deduplicated for split, count in deduplicated_counts.items()}
    
    LOGGER.info("Original split proportions (before deduplication): %s", 
                {split: f"{prop*100:.2f}%" for split, prop in proportions_before_dedup.items()})
    LOGGER.info("Split proportions after deduplication: %s", 
                {split: f"{prop*100:.2f}%" for split, prop in proportions_deduplicated.items()})
    
    # Log duplicate samples summary
    total_duplicates = sum(len(dups) for dups in duplicates_by_split.values())
    if total_duplicates > 0:
        LOGGER.info("Total duplicate samples removed: %d", total_duplicates)
        for split, duplicate_rows in duplicates_by_split.items():
            if duplicate_rows:
                LOGGER.info("  %s: %d duplicate samples", split, len(duplicate_rows))
    
    # Log skipped samples summary
    total_skipped = sum(len(skipped) for skipped in skipped_by_split.values())
    if total_skipped > 0:
        LOGGER.info("Total skipped samples (empty sequences): %d", total_skipped)
        for split, skipped_rows in skipped_by_split.items():
            if skipped_rows:
                LOGGER.info("  %s: %d skipped samples", split, len(skipped_rows))

    with tempfile.TemporaryDirectory(prefix="cdhit_resplit_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        combined_fasta = tmp_dir / "combined_prefixed.fasta"
        _write_prefixed_fasta(combined_order, combined_records, combined_fasta)

        output_prefix = tmp_dir / "combined_clustering"
        clstr_path = _run_cd_hit_est(
            input_fasta=combined_fasta,
            output_prefix=output_prefix,
            threshold=threshold,
            threads=threads,
            memory_limit=memory_limit,
            word_size=word_size,
            binary=cd_hit_binary,
        )

        # Save the .clstr file to output directory for future reuse
        saved_clstr_path = output_dir / f"clustering_t{int(threshold*100)}.clstr"
        shutil.copy2(clstr_path, saved_clstr_path)
        LOGGER.info("Saved cd-hit-est .clstr file to %s", saved_clstr_path)

        clusters = _parse_clusters(clstr_path)
        LOGGER.info("Parsed %d clusters from cd-hit output.", len(clusters))

        # Save parsed clusters for future reuse (pickle format)
        clusters_pickle_path = output_dir / f"clusters_t{int(threshold*100)}.pkl"
        _save_clusters(clusters, clusters_pickle_path)

        # Save parsed clusters in JSON format for human readability
        clusters_json_path = output_dir / f"clusters_t{int(threshold*100)}.json"
        _save_clusters_json(clusters, clusters_json_path)

        # Use ORIGINAL counts (before deduplication) as target for reassignment
        # This preserves the original split proportions
        LOGGER.info("Using original counts (before deduplication) as target for reassignment: %s", original_counts_before_dedup)
        final_counts, cluster_assignments = _assign_clusters(
            clusters, combined_records, original_counts_before_dedup
        )

        # 记录与目标计数的绝对偏差，方便快速核对
        delta_counts = {
            split: final_counts.get(split, 0) - original_counts_before_dedup.get(split, 0)
            for split in original_counts_before_dedup.keys()
        }
        LOGGER.info("Absolute deltas from target counts: %s", delta_counts)

        output_paths = _write_split_csvs(
            output_dir=output_dir,
            cluster_assignments=cluster_assignments,
            records=combined_records,
            headers=headers,
        )

        # Write detailed report
        detailed_report_path = output_dir / "resplit_detailed_report.txt"
        _write_detailed_report(
            detailed_report_path,
            cluster_assignments,
            combined_records,
            skipped_by_split,
            duplicates_by_split,
            original_counts_before_dedup,
            deduplicated_counts,
            final_counts,
        )
        LOGGER.info("Detailed report written to %s", detailed_report_path)

        if cluster_report:
            with cluster_report.open("w", encoding="utf-8") as f:
                f.write("cluster_index\tassigned_split\tcluster_size\tmember_ids\n")
                for cluster, assigned_split in cluster_assignments:
                    member_ids = ", ".join(cluster.members)
                    f.write(
                        f"{cluster.index}\t{assigned_split}\t{len(cluster.members)}\t{member_ids}\n"
                    )

    # Organize cluster indices by split for summary
    split_to_cluster_indices: Dict[str, List[int]] = {}
    for cluster, assigned_split in cluster_assignments:
        split_to_cluster_indices.setdefault(assigned_split, [])
        split_to_cluster_indices[assigned_split].append(cluster.index)

    # Calculate final proportions
    total_final = sum(final_counts.values())
    final_proportions = {split: count / total_final for split, count in final_counts.items()}

    summary = {
        "threshold": threshold,
        "total_clusters": len(clusters),
        "original_counts_before_dedup": original_counts_before_dedup,
        "original_proportions_before_dedup": {split: f"{prop*100:.2f}%" for split, prop in proportions_before_dedup.items()},
        "deduplicated_counts": deduplicated_counts,
        "deduplicated_proportions": {split: f"{prop*100:.2f}%" for split, prop in proportions_deduplicated.items()},
        "final_counts": final_counts,
        "final_proportions": {split: f"{prop*100:.2f}%" for split, prop in final_proportions.items()},
        "proportion_deviations_from_original": {
            split: f"{(final_proportions.get(split, 0) - proportions_before_dedup.get(split, 0))*100:+.2f}%"
            for split in original_counts_before_dedup.keys()
        },
        "proportion_deviations_from_deduplicated": {
            split: f"{(final_proportions.get(split, 0) - proportions_deduplicated.get(split, 0))*100:+.2f}%"
            for split in deduplicated_counts.keys()
        },
        "duplicate_counts": {split: len(dups) for split, dups in duplicates_by_split.items()},
        "skipped_counts": {split: len(skipped) for split, skipped in skipped_by_split.items()},
        "cluster_distribution": {
            split: {
                "num_clusters": len(indices),
                "cluster_indices": sorted(indices),
            }
            for split, indices in split_to_cluster_indices.items()
        },
        "output_csvs": {split: str(path) for split, path in output_paths.items()},
        "detailed_report": str(detailed_report_path),
        "saved_clstr_file": str(saved_clstr_path),
        "saved_clusters_pickle": str(clusters_pickle_path),
        "saved_clusters_json": str(clusters_json_path),
    }

    LOGGER.info("Final split sizes: %s", final_counts)
    LOGGER.info("Final split proportions: %s", 
                {split: f"{prop*100:.2f}%" for split, prop in final_proportions.items()})
    LOGGER.info("Proportion deviations from ORIGINAL (before deduplication): %s",
                {split: f"{(final_proportions.get(split, 0) - proportions_before_dedup.get(split, 0))*100:+.2f}%"
                 for split in original_counts_before_dedup.keys()})
    if any(count > 0 for count in deduplicated_counts.values()):
        LOGGER.info("Proportion deviations from deduplicated: %s",
                    {split: f"{(final_proportions.get(split, 0) - proportions_deduplicated.get(split, 0))*100:+.2f}%"
                     for split in deduplicated_counts.keys() if deduplicated_counts[split] > 0})
    
    return summary


def _parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster combined datasets with cd-hit-est and reassign clusters to "
            "maintain original split ratios."
        )
    )
    parser.add_argument("--train", type=Path, help="Path to the training CSV file.")
    parser.add_argument("--validation", type=Path, help="Path to the validation CSV file.")
    parser.add_argument("--test", type=Path, help="Path to the test CSV file.")
    parser.add_argument(
        "--sequence-column",
        type=str,
        required=True,
        help="Name of the column containing DNA/RNA sequences.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        help="Name of the column containing unique identifiers (optional).",
    )
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
        help="Directory to store the resplit CSV files and optional reports.",
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
        train_csv=parsed.train,
        validation_csv=parsed.validation,
        test_csv=parsed.test,
        sequence_column=parsed.sequence_column,
        id_column=parsed.id_column,
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