"""Utility to analyze dataset similarity against a training set using cd-hit-est.

This module provides a convenience wrapper around the `cd-hit-est-2d` executable so
that downstream workflows can quickly quantify how many sequences in the validation
and test splits are highly similar to entries in the training split under several
identity thresholds.

Typical usage from the command line:: 使用方法

    cd /home/sw1136/OmniGenBench/examples/dingling_te

python cdhit_similarity.py \
--train train.csv \
--validation valid.csv \
--test test.csv \
--thresholds 0.80 0.85 0.90 0.95 \
--threads 4 \
--output-dir cdhit_similarity_out \
--summary-json cdhit_similarity_out/summary.json

The script will generate per-threshold reports for 80%, 85%, 90%, and 95%
sequence-identity cut-offs by default. Each report includes the raw `cd-hit`
outputs along with helper files listing the identifiers from the validation/test
sets that overlap the training data.
"""

from __future__ import annotations

import argparse
import pandas as pd
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)

# Default thresholds requested by the user.
DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)


@dataclass
class CDHitRunResult:
    """Structured result for a single cd-hit-est-2d execution."""

    threshold: float
    matched_ids: List[str]
    unmatched_ids: List[str]
    total_sequences: int
    clusters_with_overlap: int
    output_prefix: Path

    def to_summary(self) -> Dict[str, object]:
        """Convert the run result into a plain-JSON friendly summary."""

        matched = len(self.matched_ids)
        return {
            "threshold": self.threshold,
            "target_sequences_total": self.total_sequences,
            "target_sequences_matched": matched,
            "target_sequences_unmatched": len(self.unmatched_ids),
            "target_match_ratio": round(matched / self.total_sequences, 4)
            if self.total_sequences
            else 0.0,
            "clusters_with_train_overlap": self.clusters_with_overlap,
            "output_files": {
                "cluster_fasta": str(self.output_prefix),
                "cluster_report": f"{self.output_prefix}.clstr",
                "matched_ids": f"{self.output_prefix}_matched_ids.txt",
                "unmatched_ids": f"{self.output_prefix}_unique_ids.txt",
            },
        }


def csv_to_fasta(
    csv_path: Path,
    fasta_path: Path,
    id_col: str = "ID",
    seq_col: str = "seq",
) -> int:
    """Convert a CSV file with sequence columns into a FASTA file.

    Parameters
    ----------
    csv_path
        Path to the CSV file containing at least ``id_col`` and ``seq_col``.
    fasta_path
        Destination FASTA file to write.
    id_col
        Column name holding unique sequence identifiers. Defaults to ``"ID"``.
    seq_col
        Column name holding raw nucleotide sequences. Defaults to ``"seq"``.

    Returns
    -------
    int
        The number of sequences written to the FASTA file.
    """

    df = pd.read_csv(csv_path)
    df = df[df[seq_col].notna() & (df[seq_col] != "")]
    count = 0
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            sid = str(row[id_col])
            seq = str(row[seq_col]).replace(" ", "").replace("\n", "")
            f.write(f">{sid}\n{seq}\n")
            count += 1
    LOGGER.info("Wrote %d sequences → %s", count, fasta_path)
    return count


def _ensure_executable(binary: str, friendly_name: str) -> None:
    """Raise an informative error if the required cd-hit binary is missing."""

    if shutil.which(binary) is None:
        raise FileNotFoundError(
            f"Required executable '{binary}' for {friendly_name} not found in PATH. "
            "Please install CD-HIT and ensure the binary is accessible."
        )


def _guess_word_size(threshold: float) -> int:
    """Infer an appropriate `-n` word size for cd-hit-est given a threshold.

    The mapping follows the guidance from the CD-HIT documentation for DNA/RNA
    sequences. Users can override this by supplying the ``--word-size`` argument.
    """

    if threshold >= 0.95:
        return 10
    if threshold >= 0.90:
        return 9
    if threshold >= 0.88:
        return 8
    if threshold >= 0.85:
        return 7
    if threshold >= 0.80:
        return 6
    return 5


def _rewrite_fasta_with_prefix(
    src: Path, dst: Path, prefix: str
) -> Tuple[int, Dict[str, str]]:
    """Rewrite a FASTA file while prefixing identifiers.

    Parameters
    ----------
    src:
        Original FASTA file.
    dst:
        Destination file with prefixed identifiers.
    prefix:
        Prefix to add to each sequence identifier. The prefix is only attached to
        the primary identifier token (before the first whitespace).

    Returns
    -------
    tuple
        A pair of ``(sequence_count, id_mapping)`` where ``id_mapping`` maps the
        prefixed identifier to the original identifier (without description).
    """

    count = 0
    id_mapping: Dict[str, str] = {}
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith(">"):
                count += 1
                header = line[1:].strip()
                if header:
                    header_parts = header.split(maxsplit=1)
                    original_id = header_parts[0]
                    description = header_parts[1] if len(header_parts) > 1 else ""
                else:
                    original_id = f"unnamed_{count}"
                    description = ""

                prefixed_id = f"{prefix}{original_id}"
                id_mapping[prefixed_id] = original_id
                if description:
                    fout.write(f">{prefixed_id} {description}\n")
                else:
                    fout.write(f">{prefixed_id}\n")
            else:
                fout.write(line)
    return count, id_mapping


def _extract_identifier_from_clstr(line: str) -> str:
    """Extract the sequence identifier from a cd-hit `.clstr` entry line."""

    start = line.find(">")
    end = line.find("...", start)
    if start == -1 or end == -1:
        raise ValueError(
            "Unexpected cd-hit cluster line format; unable to locate identifier."
        )
    return line[start + 1 : end]


def _parse_clstr_file(
    clstr_path: Path, train_prefix: str, target_prefix: str
) -> Tuple[int, List[str]]:
    """Parse the cd-hit cluster report to recover target IDs matching the train set.

    Returns
    -------
    tuple
        ``(clusters_with_overlap, matched_ids)`` where ``matched_ids`` retains the
        prefixed identifiers (target sequences) that cluster with at least one
        training sequence.
    """

    clusters_with_overlap = 0
    matched_target_ids: List[str] = []

    has_train = False
    target_ids_in_cluster: List[str] = []

    with clstr_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                # Finalize the previous cluster before resetting.
                if has_train and target_ids_in_cluster:
                    clusters_with_overlap += 1
                    matched_target_ids.extend(target_ids_in_cluster)
                has_train = False
                target_ids_in_cluster = []
                continue

            seq_id = _extract_identifier_from_clstr(line)
            if seq_id.startswith(train_prefix):
                has_train = True
            elif seq_id.startswith(target_prefix):
                target_ids_in_cluster.append(seq_id)

        # Final cluster flush
        if has_train and target_ids_in_cluster:
            clusters_with_overlap += 1
            matched_target_ids.extend(target_ids_in_cluster)

    return clusters_with_overlap, matched_target_ids


def _write_identifier_list(path: Path, identifiers: Iterable[str]) -> None:
    """Persist identifiers to a text file, one per line."""

    with path.open("w", encoding="utf-8") as f:
        for identifier in identifiers:
            f.write(f"{identifier}\n")


def _run_cd_hit_est_2d(
    cd_hit_binary: str,
    train_fasta: Path,
    target_fasta: Path,
    output_prefix: Path,
    threshold: float,
    word_size: Optional[int] = None,
    threads: int = 4,
    memory_limit: int = 0,
) -> None:
    """Execute the cd-hit-est-2d command with the provided parameters."""

    _ensure_executable(cd_hit_binary, "cd-hit-est-2d comparisons")

    if word_size is None:
        word_size = _guess_word_size(threshold)

    cmd = [
        cd_hit_binary,
        "-i",
        str(train_fasta),
        "-i2",
        str(target_fasta),
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

    LOGGER.info(
        "Running %s", " ".join(str(part) for part in cmd)
    )
    subprocess.run(cmd, check=True)


def _derive_result(
    *,
    clstr_path: Path,
    target_prefix: str,
    prefix_mapping: Dict[str, str],
    total_sequences: int,
    threshold: float,
    output_prefix: Path,
) -> CDHitRunResult:
    """Generate a ``CDHitRunResult`` data object from a cd-hit run."""

    clusters_with_overlap, matched_prefixed_ids = _parse_clstr_file(
        clstr_path, train_prefix="train|", target_prefix=target_prefix
    )

    matched_original_ids_set = {
        prefix_mapping.get(seq_id, seq_id[len(target_prefix) :])
        for seq_id in matched_prefixed_ids
    }

    unmatched_ids = [
        original_id
        for original_id in prefix_mapping.values()
        if original_id not in matched_original_ids_set
    ]

    return CDHitRunResult(
        threshold=threshold,
        matched_ids=sorted(matched_original_ids_set),
        unmatched_ids=sorted(unmatched_ids),
        total_sequences=total_sequences,
        clusters_with_overlap=clusters_with_overlap,
        output_prefix=output_prefix,
    )


def analyze_similarity(
    *,
    train_fasta: Path,
    validation_fasta: Optional[Path] = None,
    test_fasta: Optional[Path] = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    output_dir: Path = Path("cdhit_similarity"),
    cd_hit_binary: str = "cd-hit-est-2d",
    threads: int = 4,
    memory_limit: int = 0,
    word_size: Optional[int] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Run cd-hit-est-2d comparisons for the requested datasets.

    Parameters
    ----------
    train_fasta
        FASTA file containing the training sequences.
    validation_fasta
        FASTA file containing the validation sequences. When provided, similarity
        against the training set will be computed and reported.
    test_fasta
        FASTA file containing the test sequences. When provided, similarity
        against the training set will be computed and reported.
    thresholds
        Iterable of sequence-identity thresholds to evaluate.
    output_dir
        Directory to store cd-hit outputs and auxiliary reports.
    cd_hit_binary
        Path to the ``cd-hit-est-2d`` executable (defaults to resolving from PATH).
    threads
        Number of CPU threads to allocate for cd-hit.
    memory_limit
        Maximum memory (MB) cd-hit is allowed to allocate. ``0`` means no limit.
    word_size
        Optional override for the ``-n`` word size parameter. When ``None`` the
        value is inferred from the identity threshold.

    Returns
    -------
    dict
        Mapping containing summaries for each dataset split that was analyzed.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fasta = Path(train_fasta)
    validation_fasta = Path(validation_fasta) if validation_fasta else None
    test_fasta = Path(test_fasta) if test_fasta else None

    summaries: Dict[str, List[Dict[str, object]]] = {}

    with tempfile.TemporaryDirectory(prefix="cdhit_pref_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # If inputs are CSV, convert to FASTA on the fly in the temp directory.
        train_input_fa = train_fasta
        if train_input_fa.suffix.lower() == ".csv":
            train_input_fa = tmp_dir / "train_from_csv.fasta"
            csv_to_fasta(train_fasta, train_input_fa, id_col="ID", seq_col="seq")

        train_pref = tmp_dir / "train_prefixed.fasta"
        _rewrite_fasta_with_prefix(train_input_fa, train_pref, "train|")

        dataset_inputs: List[Tuple[str, Path, Dict[str, str], int]] = []

        if validation_fasta is not None:
            # Convert CSV to FASTA if needed
            val_input_fa = validation_fasta
            if val_input_fa.suffix.lower() == ".csv":
                val_input_fa = tmp_dir / "validation_from_csv.fasta"
                csv_to_fasta(validation_fasta, val_input_fa, id_col="ID", seq_col="seq")

            val_pref = tmp_dir / "validation_prefixed.fasta"
            val_count, val_mapping = _rewrite_fasta_with_prefix(
                val_input_fa, val_pref, "validation|"
            )
            dataset_inputs.append(("validation", val_pref, val_mapping, val_count))

        if test_fasta is not None:
            # Convert CSV to FASTA if needed
            test_input_fa = test_fasta
            if test_input_fa.suffix.lower() == ".csv":
                test_input_fa = tmp_dir / "test_from_csv.fasta"
                csv_to_fasta(test_fasta, test_input_fa, id_col="ID", seq_col="seq")

            test_pref = tmp_dir / "test_prefixed.fasta"
            test_count, test_mapping = _rewrite_fasta_with_prefix(
                test_input_fa, test_pref, "test|"
            )
            dataset_inputs.append(("test", test_pref, test_mapping, test_count))

        for dataset_name, pref_path, id_mapping, total_sequences in dataset_inputs:
            dataset_results: List[Dict[str, object]] = []

            if total_sequences == 0:
                LOGGER.warning(
                    "Dataset '%s' contains no sequences; skipping cd-hit analysis.",
                    dataset_name,
                )
                summaries[dataset_name] = []
                continue

            for threshold in thresholds:
                output_prefix = (
                    output_dir
                    / f"{dataset_name}_vs_train_{int(threshold * 100):02d}"
                )
                _run_cd_hit_est_2d(
                    cd_hit_binary=cd_hit_binary,
                    train_fasta=train_pref,
                    target_fasta=pref_path,
                    output_prefix=output_prefix,
                    threshold=threshold,
                    word_size=word_size,
                    threads=threads,
                    memory_limit=memory_limit,
                )

                clstr_path = Path(f"{output_prefix}.clstr")
                result = _derive_result(
                    clstr_path=clstr_path,
                    target_prefix=f"{dataset_name}|",
                    prefix_mapping=id_mapping,
                    total_sequences=total_sequences,
                    threshold=threshold,
                    output_prefix=output_prefix,
                )

                _write_identifier_list(
                    Path(f"{output_prefix}_matched_ids.txt"), result.matched_ids
                )
                _write_identifier_list(
                    Path(f"{output_prefix}_unique_ids.txt"), result.unmatched_ids
                )

                dataset_results.append(result.to_summary())

            summaries[dataset_name] = dataset_results

    return summaries


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def _parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze validation/test similarity to the training set across "
            "multiple cd-hit-est thresholds."
        )
    )
    parser.add_argument(
        "--train",
        required=True,
        type=Path,
        help=(
            "Path to the training file. Supports FASTA or CSV (with columns 'ID' and 'seq')."
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        help=(
            "Path to the validation file (optional). Supports FASTA or CSV (with columns 'ID' and 'seq')."
        ),
    )
    parser.add_argument(
        "--test",
        type=Path,
        help=(
            "Path to the test file (optional). Supports FASTA or CSV (with columns 'ID' and 'seq')."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help=(
            "Sequence-identity thresholds to evaluate (default: 0.80 0.85 0.90 0.95)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cdhit_similarity"),
        help="Directory where cd-hit outputs and summaries will be stored.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for cd-hit (default: 4).",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=0,
        help="Maximum memory in MB for cd-hit (default: 0 meaning unlimited).",
    )
    parser.add_argument(
        "--word-size",
        type=int,
        help="Override the cd-hit word size (-n). If omitted, inferred automatically.",
    )
    parser.add_argument(
        "--cd-hit-binary",
        default="cd-hit-est-2d",
        help="Name or path of the cd-hit-est-2d executable.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to save the aggregated similarity summary as JSON.",
    )

    parsed_args = parser.parse_args(args=args)
    if not parsed_args.validation and not parsed_args.test:
        parser.error("At least one of --validation or --test must be provided.")

    return parsed_args


def main(args: Optional[Sequence[str]] = None) -> Dict[str, List[Dict[str, object]]]:
    parsed_args = _parse_args(args)
    _configure_logging(parsed_args.log_level)

    summaries = analyze_similarity(
        train_fasta=parsed_args.train,
        validation_fasta=parsed_args.validation,
        test_fasta=parsed_args.test,
        thresholds=parsed_args.thresholds,
        output_dir=parsed_args.output_dir,
        cd_hit_binary=parsed_args.cd_hit_binary,
        threads=parsed_args.threads,
        memory_limit=parsed_args.memory_limit,
        word_size=parsed_args.word_size,
    )

    if parsed_args.summary_json:
        with parsed_args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        LOGGER.info("Summary written to %s", parsed_args.summary_json)

    LOGGER.info("Similarity analysis completed.")
    return summaries


if __name__ == "__main__":
    main()
