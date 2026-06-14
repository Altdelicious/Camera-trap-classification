from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class ExperimentConfig:
    domain_paths: Dict[str, Path]
    transfer_pairs: List[Tuple[str, str]]
    taxonomy_mapping_path: Path | None = None
    taxonomy_domain_keys: Dict[str, str] | None = None
    taxonomy_dataset_column: str | None = None
    taxonomy_label_column: str | None = None
    taxonomy_target_column: str | None = None
    n_blocks: int = 5
    k_factor: float = 3.0
    init_train_ratio: float = 0.8
    output_dir: Path = Path("outputs")


@dataclass
class Dataset:
    embeddings: np.ndarray
    species: np.ndarray
    timestamps: np.ndarray
    dates: np.ndarray
    locations: np.ndarray


@dataclass
class DomainData:
    name: str
    dataset: Dataset
    blocks: List[np.ndarray]


@dataclass
class TaxonomyMapping:
    dataset_column: str
    label_column: str
    target_column: str
    label_map: Dict[Tuple[str, str], str]


def _decode_bytes(values: Sequence[object]) -> np.ndarray:
    return np.array(
        [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]
    )


def _norm_key(value: object) -> str:
    return str(value).strip().casefold()


def _first_matching_column(
    columns: Sequence[str],
    explicit: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"Taxonomy mapping {role} column '{explicit}' was not found. "
                f"Available columns: {list(columns)}"
            )
        return explicit

    normalized = {_norm_key(col): col for col in columns}
    for candidate in candidates:
        match = normalized.get(_norm_key(candidate))
        if match is not None:
            return match

    raise ValueError(
        f"Could not infer taxonomy mapping {role} column. "
        f"Available columns: {list(columns)}"
    )


def load_taxonomy_mapping(config: ExperimentConfig) -> TaxonomyMapping | None:
    if config.taxonomy_mapping_path is None:
        return None

    mapping_df = pd.read_csv(config.taxonomy_mapping_path)
    columns = list(mapping_df.columns)

    dataset_column = _first_matching_column(
        columns,
        config.taxonomy_dataset_column,
        ["dataset_name", "dataset", "ds_name", "short_name"],
        "dataset",
    )
    label_column = _first_matching_column(
        columns,
        config.taxonomy_label_column,
        ["original_label", "category", "label", "source_label"],
        "original label",
    )

    target_candidates = []
    if config.taxonomy_target_column is not None:
        target_candidates.append(config.taxonomy_target_column)
    target_candidates.extend(
        [
            "scientific_name",
            "common_name",
            "query",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
        ]
    )
    target_column = _first_matching_column(
        columns,
        config.taxonomy_target_column,
        target_candidates,
        "target taxonomy",
    )

    valid_rows = mapping_df[[dataset_column, label_column, target_column]].dropna(
        subset=[dataset_column, label_column, target_column]
    )

    label_map: Dict[Tuple[str, str], str] = {}
    for _, row in valid_rows.iterrows():
        target_value = str(row[target_column]).strip()
        if not target_value:
            continue
        key = (_norm_key(row[dataset_column]), _norm_key(row[label_column]))
        label_map[key] = target_value

    print("=" * 72)
    print("TAXONOMY MAPPING")
    print(f"  Path            : {config.taxonomy_mapping_path}")
    print(f"  Dataset column  : {dataset_column}")
    print(f"  Label column    : {label_column}")
    print(f"  Target column   : {target_column}")
    print(f"  Usable mappings : {len(label_map):,}")

    return TaxonomyMapping(
        dataset_column=dataset_column,
        label_column=label_column,
        target_column=target_column,
        label_map=label_map,
    )


def harmonize_species_labels(
    species: np.ndarray,
    domain_name: str,
    taxonomy_mapping: TaxonomyMapping | None,
    taxonomy_domain_keys: Dict[str, str] | None,
) -> np.ndarray:
    if taxonomy_mapping is None:
        return species

    dataset_key = (
        taxonomy_domain_keys.get(domain_name, domain_name)
        if taxonomy_domain_keys is not None
        else domain_name
    )
    dataset_norm = _norm_key(dataset_key)

    remapped = []
    n_mapped = 0
    unmapped_labels = set()
    for label in species:
        mapped = taxonomy_mapping.label_map.get((dataset_norm, _norm_key(label)))
        if mapped is None:
            remapped.append(label)
            unmapped_labels.add(str(label))
        else:
            remapped.append(mapped)
            if mapped != label:
                n_mapped += 1

    remapped_arr = np.asarray(remapped)
    print(f"  Taxonomy key    : {dataset_key}")
    print(f"  Labels remapped : {n_mapped:,}")
    print(f"  Unique taxa     : {len(np.unique(remapped_arr))}")
    if unmapped_labels:
        preview = ", ".join(sorted(unmapped_labels)[:8])
        extra = "" if len(unmapped_labels) <= 8 else ", ..."
        print(
            f"  Unmapped labels : {len(unmapped_labels)} "
            f"(kept as original labels: {preview}{extra})"
        )

    return remapped_arr


def load_dataset(
    h5_path: Path,
    domain_name: str,
    taxonomy_mapping: TaxonomyMapping | None = None,
    taxonomy_domain_keys: Dict[str, str] | None = None,
) -> Dataset:
    with h5py.File(h5_path, "r") as hf:
        raw_embeddings = hf["embeddings"][:]
        raw_species = _decode_bytes(hf["species"][:])
        raw_strings = _decode_bytes(hf["date_captured"][:])
        if "location" in hf:
            raw_locations = _decode_bytes(hf["location"][:])
        else:
            raw_locations = np.full(len(raw_embeddings), "", dtype=object)

    temp_times = pd.to_datetime(raw_strings, errors="coerce")
    valid_timestamps = np.asarray(pd.notna(temp_times))
    finite_embeddings = np.isfinite(raw_embeddings).all(axis=1)
    nonzero_embeddings = np.linalg.norm(raw_embeddings, axis=1) > 0
    mask = valid_timestamps & finite_embeddings & nonzero_embeddings

    dropped = int((~mask).sum())
    if not np.any(mask):
        raise ValueError(
            f"Domain '{domain_name}' has no valid samples after cleaning timestamps and embeddings."
        )

    embeddings = normalize(raw_embeddings[mask], norm="l2")
    species = raw_species[mask]
    timestamps = np.asarray(temp_times)[mask]
    locations = raw_locations[mask]

    sort_idx = np.argsort(timestamps)
    embeddings = embeddings[sort_idx]
    species = species[sort_idx]
    timestamps = timestamps[sort_idx]
    locations = locations[sort_idx]
    dates = timestamps.astype("datetime64[D]")
    species = harmonize_species_labels(
        species, domain_name, taxonomy_mapping, taxonomy_domain_keys
    )

    print("=" * 72)
    print(f"DOMAIN: {domain_name}")
    print(f"  Path            : {h5_path}")
    print(f"  Samples loaded  : {len(embeddings):,}")
    print(f"  Samples dropped : {dropped:,}")
    print(f"  Time span       : {timestamps.min()} -> {timestamps.max()}")
    print(f"  Unique species  : {len(np.unique(species))}")

    return Dataset(
        embeddings=embeddings,
        species=species,
        timestamps=timestamps,
        dates=dates,
        locations=locations,
    )


def make_equal_time_blocks(dates: np.ndarray, n_blocks: int) -> List[np.ndarray]:
    if n_blocks < 1:
        raise ValueError("n_blocks must be at least 1.")
    if len(dates) == 0:
        raise ValueError("Cannot create blocks for an empty domain.")
    if len(dates) < n_blocks:
        raise ValueError(
            f"Cannot split {len(dates)} samples into {n_blocks} non-empty blocks."
        )

    date_days = dates.astype("datetime64[D]").astype(np.int64)
    unique_days = np.unique(date_days)
    if len(unique_days) < n_blocks:
        raise ValueError(
            f"Only {len(unique_days)} unique dates are available, fewer than {n_blocks} blocks."
        )

    start_day = int(unique_days[0])
    end_day = int(unique_days[-1])
    span_days = end_day - start_day + 1
    if span_days < n_blocks:
        raise ValueError(
            f"Time span covers only {span_days} calendar days, fewer than {n_blocks} blocks."
        )

    boundaries = np.floor(np.linspace(0, span_days, num=n_blocks + 1)).astype(int)
    if np.any(np.diff(boundaries) == 0):
        raise ValueError(
            f"Temporal boundaries collapsed for a {span_days}-day span and {n_blocks} blocks."
        )

    day_to_block: Dict[int, int] = {}
    for day in unique_days:
        offset = int(day - start_day)
        block_id = int(np.searchsorted(boundaries[1:], offset, side="right"))
        day_to_block[int(day)] = min(block_id, n_blocks - 1)

    blocks: List[List[int]] = [[] for _ in range(n_blocks)]
    for idx, day in enumerate(date_days):
        blocks[day_to_block[int(day)]].append(idx)

    result = [np.asarray(block, dtype=int) for block in blocks]
    empty_blocks = [i + 1 for i, block in enumerate(result) if len(block) == 0]
    if empty_blocks:
        raise ValueError(
            "Equal-time blocking produced empty blocks "
            f"{empty_blocks}. The domain likely has large temporal gaps."
        )

    return result


def print_blocks(label: str, blocks: Sequence[np.ndarray], dates: np.ndarray) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {label} ({len(blocks)} blocks)")
    print(f"  {'Block':>6}  {'Size':>7}  {'Date range'}")
    print(f"  {'-' * 52}")
    for i, block in enumerate(blocks, start=1):
        d_min = dates[block].min()
        d_max = dates[block].max()
        print(f"  {i:>6}  {len(block):>7,}  {d_min} -> {d_max}")


def build_domain(
    name: str,
    h5_path: Path,
    n_blocks: int,
    taxonomy_mapping: TaxonomyMapping | None = None,
    taxonomy_domain_keys: Dict[str, str] | None = None,
) -> DomainData:
    dataset = load_dataset(h5_path, name, taxonomy_mapping, taxonomy_domain_keys)
    blocks = make_equal_time_blocks(dataset.dates, n_blocks)
    print_blocks(name, blocks, dataset.dates)

    total = sum(len(block) for block in blocks)
    if total != len(dataset.embeddings):
        raise ValueError(
            f"Domain '{name}' block sample mismatch: got {total:,}, expected {len(dataset.embeddings):,}."
        )
    print(f"  Total samples across blocks: {total:,}")

    return DomainData(name=name, dataset=dataset, blocks=blocks)


class AdaptiveNCMClassifier:
    def __init__(self, k_factor: float = 2.0, default_radius: float = 0.15):
        self.k_factor = k_factor
        self.default_radius = default_radius
        self.prototypes: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self.mean_dists: Dict[str, float] = {}
        self.m2_dists: Dict[str, float] = {}

    @staticmethod
    def _normalized(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_radius(self, cls: str) -> float:
        n = self.counts[cls]
        if n < 2:
            return self.default_radius

        variance = max(0.0, self.m2_dists[cls] / (n - 1))
        std = np.sqrt(variance)
        return self.mean_dists[cls] + max(0.05, self.k_factor * std)

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_embs = embeddings[cls_mask]
            n = len(cls_embs)

            proto = np.mean(cls_embs, axis=0)
            proto_norm = self._normalized(proto)
            dots = np.clip(cls_embs.dot(proto_norm), -1.0, 1.0)
            dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))

            self.prototypes[cls] = proto
            self.counts[cls] = n
            self.mean_dists[cls] = float(np.mean(dists))
            self.m2_dists[cls] = float(np.sum((dists - self.mean_dists[cls]) ** 2))

    def update(self, new_emb: np.ndarray, species: str) -> None:
        n = self.counts.get(species, 0)
        if n == 0:
            self.prototypes[species] = new_emb.copy()
            self.counts[species] = 1
            self.mean_dists[species] = 0.0
            self.m2_dists[species] = 0.0
            return

        old_proto = self.prototypes[species]
        new_n = n + 1
        self.prototypes[species] = (old_proto * n + new_emb) / new_n

        old_proto_norm = self._normalized(old_proto)
        emb_norm = self._normalized(new_emb)
        dot = np.clip(np.dot(old_proto_norm, emb_norm), -1.0, 1.0)
        new_dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * dot)))

        delta = new_dist - self.mean_dists[species]
        self.mean_dists[species] += delta / new_n
        delta2 = new_dist - self.mean_dists[species]
        self.m2_dists[species] += delta * delta2
        self.counts[species] = new_n

    def predict(self, query_emb: np.ndarray) -> Tuple[str | None, bool]:
        if not self.prototypes:
            return None, True

        species_list = list(self.prototypes.keys())
        query_norm = self._normalized(query_emb)
        proto_matrix = np.array(
            [self._normalized(self.prototypes[cls]) for cls in species_list]
        )

        dots = np.clip(proto_matrix.dot(query_norm), -1.0, 1.0)
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))

        within_radius = [
            i
            for i, (cls, dist) in enumerate(zip(species_list, dists))
            if dist <= self._get_radius(cls)
        ]

        if not within_radius:
            nearest_idx = int(np.argmin(dists))
            return species_list[nearest_idx], True

        best_idx = min(within_radius, key=lambda i: dists[i])
        return species_list[best_idx], False

    @property
    def known_species(self) -> set:
        return set(self.prototypes.keys())


def evaluate_block(
    clf: AdaptiveNCMClassifier, embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, object]:
    known_at_eval = clf.known_species.copy()
    sp_correct = {sp: 0 for sp in known_at_eval}
    sp_total = {sp: 0 for sp in known_at_eval}
    correct = 0
    new_species_total = 0
    correctly_flagged_new = 0
    known_but_flagged_unknown = 0
    falsely_accepted_new = 0

    for emb, lbl in zip(embeddings, labels):
        pred, is_unknown = clf.predict(emb)
        truly_new = lbl not in known_at_eval

        if truly_new:
            new_species_total += 1
            if is_unknown:
                correctly_flagged_new += 1
            else:
                falsely_accepted_new += 1
            continue

        sp_total[lbl] += 1
        if is_unknown:
            known_but_flagged_unknown += 1
        elif pred == lbl:
            correct += 1
            sp_correct[lbl] += 1

    n_known_samples = sum(sp_total.values())
    accuracy = correct / n_known_samples if n_known_samples > 0 else float("nan")
    per_species_accuracy = {
        sp: (sp_correct[sp] / sp_total[sp] if sp_total[sp] > 0 else float("nan"))
        for sp in known_at_eval
    }
    discovery_rate = (
        correctly_flagged_new / new_species_total
        if new_species_total > 0
        else float("nan")
    )
    false_rejection_rate = (
        known_but_flagged_unknown / n_known_samples if n_known_samples > 0 else 0.0
    )
    false_acceptance_rate = (
        falsely_accepted_new / new_species_total if new_species_total > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "per_species_accuracy": per_species_accuracy,
        "discovery_rate": discovery_rate,
        "false_rejection_rate": false_rejection_rate,
        "false_acceptance_rate": false_acceptance_rate,
        "n_falsely_accepted": falsely_accepted_new,
        "n_new_species": new_species_total,
        "n_known_samples": n_known_samples,
        "total": len(labels),
    }


def _blk(domain: DomainData, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return domain.dataset.embeddings[indices], domain.dataset.species[indices]


def _fmt(value: float) -> str:
    return f"{value:.3f}" if not np.isnan(value) else "n/a"


def _get_seed_clf(
    blocks: Sequence[np.ndarray],
    embeddings: np.ndarray,
    species: np.ndarray,
    init_train_ratio: float,
    k_factor: float,
) -> Tuple[AdaptiveNCMClassifier, np.ndarray]:
    block0_idx = blocks[0]
    n_init = max(1, int(init_train_ratio * len(block0_idx)))
    if n_init >= len(block0_idx):
        raise ValueError("Initial train ratio leaves no hold-out samples in block 1.")

    init_idx = block0_idx[:n_init]
    hold_idx = block0_idx[n_init:]

    clf = AdaptiveNCMClassifier(k_factor=k_factor)
    clf.fit(embeddings[init_idx], species[init_idx])
    return clf, hold_idx


def run_internal_pipeline(
    domain: DomainData,
    config: ExperimentConfig,
    update_model: bool = True,
) -> List[Dict[str, object]]:
    clf, hold_idx = _get_seed_clf(
        domain.blocks,
        domain.dataset.embeddings,
        domain.dataset.species,
        config.init_train_ratio,
        config.k_factor,
    )

    print("=" * 70)
    print(
        f"{domain.name} INTERNAL - {'ADAPTIVE' if update_model else 'STATIC'}"
    )
    print("=" * 70)

    results: List[Dict[str, object]] = []

    hold_embs, hold_lbls = _blk(domain, hold_idx)
    n_before = len(clf.known_species)
    metrics = evaluate_block(clf, hold_embs, hold_lbls)

    if update_model:
        for emb, lbl in zip(hold_embs, hold_lbls):
            clf.update(emb, lbl)

    n_after = len(clf.known_species)
    results.append(
        {
            "block": 1,
            **metrics,
            "n_known_before": n_before,
            "n_known_after": n_after,
        }
    )
    print(
        f"  Block 1 | Acc: {_fmt(metrics['accuracy'])} | "
        f"DR: {_fmt(metrics['discovery_rate'])} | Known: {n_before} -> {n_after}"
    )

    for i, block_idx in enumerate(domain.blocks[1:], start=2):
        block_embs, block_lbls = _blk(domain, block_idx)
        n_before = len(clf.known_species)
        metrics = evaluate_block(clf, block_embs, block_lbls)

        if update_model:
            for emb, lbl in zip(block_embs, block_lbls):
                clf.update(emb, lbl)

        n_after = len(clf.known_species)
        results.append(
            {
                "block": i,
                **metrics,
                "n_known_before": n_before,
                "n_known_after": n_after,
            }
        )
        print(
            f"  Block {i} | Acc: {_fmt(metrics['accuracy'])} | "
            f"DR: {_fmt(metrics['discovery_rate'])} | Known: {n_before} -> {n_after}"
        )

    print("-" * 70)
    return results


def run_transfer_matrix(
    source_domain: DomainData,
    target_domain: DomainData,
    config: ExperimentConfig,
    update_model: bool = True,
) -> List[List[Dict[str, object]]]:
    clf, hold_idx = _get_seed_clf(
        source_domain.blocks,
        source_domain.dataset.embeddings,
        source_domain.dataset.species,
        config.init_train_ratio,
        config.k_factor,
    )

    hold_embs, hold_lbls = _blk(source_domain, hold_idx)
    for emb, lbl in zip(hold_embs, hold_lbls):
        clf.update(emb, lbl)

    print("=" * 84)
    print(
        f"TRANSFER MATRIX: {source_domain.name} -> {target_domain.name} "
        f"({'Adaptive' if update_model else 'Static'})"
    )
    print("=" * 84)
    header = " " * 18 + "".join(
        f"{target_domain.name}[{j + 1}]".rjust(12)
        for j in range(len(target_domain.blocks))
    )
    print(header)

    matrix: List[List[Dict[str, object]]] = []
    for i in range(1, len(source_domain.blocks) + 1):
        if update_model and i > 1:
            src_embs, src_lbls = _blk(source_domain, source_domain.blocks[i - 1])
            for emb, lbl in zip(src_embs, src_lbls):
                clf.update(emb, lbl)

        row_metrics: List[Dict[str, object]] = []
        row_label = (
            f"  M_{source_domain.name}[{i}] "
            f"k={len(clf.known_species):>3} |"
        )

        for tgt_idx in target_domain.blocks:
            tgt_embs, tgt_lbls = _blk(target_domain, tgt_idx)
            metrics = evaluate_block(clf, tgt_embs, tgt_lbls)
            row_metrics.append(metrics)
            row_label += f"  {_fmt(metrics['accuracy']):>5}"

        matrix.append(row_metrics)
        print(row_label)

    print("-" * 84)
    return matrix


def get_scenario_trajectories(
    matrix_adaptive: List[List[Dict[str, object]]],
    matrix_static: List[List[Dict[str, object]]],
) -> Tuple[List[float], List[float]]:
    baseline = [m["accuracy"] for m in matrix_static[0]]
    method = [m["accuracy"] for m in matrix_adaptive[-1]]
    return baseline, method


def plot_internal_accuracy(
    internal_results: Dict[str, Dict[str, List[Dict[str, object]]]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")

    for idx, (domain_name, variants) in enumerate(internal_results.items()):
        color = cmap(idx % 10)
        adaptive = variants["adaptive"]
        static = variants["static"]
        blocks = [r["block"] for r in adaptive]

        plt.plot(
            blocks,
            [r["accuracy"] for r in adaptive],
            "o-",
            color=color,
            linewidth=2,
            label=f"{domain_name} (Adaptive)",
        )
        plt.plot(
            blocks,
            [r["accuracy"] for r in static],
            "o--",
            color=color,
            alpha=0.55,
            label=f"{domain_name} (Static)",
        )

    plt.title("Internal Temporal Accuracy Across Domains", fontsize=14)
    plt.xlabel("Timeline Blocks", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(blocks)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "internal_temporal_accuracy.png", dpi=200)
    plt.close()


def plot_transfer_scenarios(
    transfer_results: Dict[Tuple[str, str], Dict[str, List[List[Dict[str, object]]]]],
    output_dir: Path,
) -> None:
    if not transfer_results:
        return

    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")

    for idx, ((src, dst), variants) in enumerate(transfer_results.items()):
        color = cmap(idx % 10)
        baseline, method = get_scenario_trajectories(
            variants["adaptive"], variants["static"]
        )
        blocks = list(range(1, len(method) + 1))
        label = f"{src} -> {dst}"

        plt.plot(
            blocks,
            method,
            "o-",
            color=color,
            linewidth=2.3,
            label=f"{label} (Adaptive)",
        )
        plt.plot(
            blocks,
            baseline,
            "o--",
            color=color,
            alpha=0.5,
            label=f"{label} (Static)",
        )

    plt.title("Selected Cross-Domain Transfer Scenarios", fontsize=14)
    plt.xlabel("Target Domain Timeline Blocks", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(blocks)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "selected_transfer_accuracy.png", dpi=200)
    plt.close()


def export_internal_results(
    internal_results: Dict[str, Dict[str, List[Dict[str, object]]]],
    output_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for domain_name, variants in internal_results.items():
        for mode, results in variants.items():
            for result in results:
                rows.append(
                    {
                        "domain": domain_name,
                        "mode": mode,
                        "block": result["block"],
                        "accuracy": result["accuracy"],
                        "discovery_rate": result["discovery_rate"],
                        "false_rejection_rate": result["false_rejection_rate"],
                        "false_acceptance_rate": result["false_acceptance_rate"],
                        "n_new_species": result["n_new_species"],
                        "n_known_samples": result["n_known_samples"],
                        "total": result["total"],
                        "n_known_before": result["n_known_before"],
                        "n_known_after": result["n_known_after"],
                    }
                )

    pd.DataFrame(rows).to_csv(output_dir / "internal_results.csv", index=False)


def export_transfer_results(
    transfer_results: Dict[Tuple[str, str], Dict[str, List[List[Dict[str, object]]]]],
    output_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for (src, dst), variants in transfer_results.items():
        for mode, matrix in variants.items():
            for source_stage, stage_results in enumerate(matrix, start=1):
                for target_block, metrics in enumerate(stage_results, start=1):
                    rows.append(
                        {
                            "source_domain": src,
                            "target_domain": dst,
                            "mode": mode,
                            "source_stage": source_stage,
                            "target_block": target_block,
                            "accuracy": metrics["accuracy"],
                            "discovery_rate": metrics["discovery_rate"],
                            "false_rejection_rate": metrics["false_rejection_rate"],
                            "false_acceptance_rate": metrics["false_acceptance_rate"],
                            "n_new_species": metrics["n_new_species"],
                            "n_known_samples": metrics["n_known_samples"],
                            "total": metrics["total"],
                        }
                    )

    pd.DataFrame(rows).to_csv(output_dir / "transfer_results.csv", index=False)


def run_experiment(config: ExperimentConfig) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = [12, 5]
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration set.")
    print(f"  Domains         : {', '.join(config.domain_paths.keys())}")
    print(f"  Blocks/domain   : {config.n_blocks}")
    print(f"  k_factor        : {config.k_factor}")
    print(f"  Init train frac : {config.init_train_ratio}")
    if config.transfer_pairs:
        pairs_str = ", ".join(f"{src}->{dst}" for src, dst in config.transfer_pairs)
        print(f"  Transfer pairs  : {pairs_str}")
    else:
        print("  Transfer pairs  : none")

    taxonomy_mapping = load_taxonomy_mapping(config)

    domains = {
        name: build_domain(
            name,
            path,
            config.n_blocks,
            taxonomy_mapping,
            config.taxonomy_domain_keys,
        )
        for name, path in config.domain_paths.items()
    }

    internal_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for name, domain in domains.items():
        internal_results[name] = {
            "adaptive": run_internal_pipeline(domain, config, True),
            "static": run_internal_pipeline(domain, config, False),
        }

    transfer_results: Dict[
        Tuple[str, str], Dict[str, List[List[Dict[str, object]]]]
    ] = {}
    for src, dst in config.transfer_pairs:
        transfer_results[(src, dst)] = {
            "adaptive": run_transfer_matrix(domains[src], domains[dst], config, True),
            "static": run_transfer_matrix(domains[src], domains[dst], config, False),
        }

    plot_internal_accuracy(internal_results, config.output_dir)
    plot_transfer_scenarios(transfer_results, config.output_dir)
    export_internal_results(internal_results, config.output_dir)
    export_transfer_results(transfer_results, config.output_dir)
    print(f"\nSaved outputs to: {config.output_dir.resolve()}")


def _parse_domain_specs(domain_specs: Sequence[str]) -> Dict[str, Path]:
    domain_paths: Dict[str, Path] = {}
    for spec in domain_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --domain value '{spec}'. Use the form name=/path/to/file.h5."
            )

        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name:
            raise ValueError(f"Invalid --domain value '{spec}': empty domain name.")
        if name in domain_paths:
            raise ValueError(f"Duplicate domain name '{name}' provided.")

        domain_paths[name] = path

    return domain_paths


def _parse_key_value_specs(specs: Sequence[str] | None, argument_name: str) -> Dict[str, str]:
    if not specs:
        return {}

    result: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid {argument_name} value '{spec}'. Use the form key=value."
            )
        key, value = spec.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid {argument_name} value '{spec}'. Use the form key=value."
            )
        result[key] = value

    return result


def _default_transfer_pairs(domain_names: Sequence[str]) -> List[Tuple[str, str]]:
    if len(domain_names) < 2:
        return []

    return [
        (domain_names[i], domain_names[(i + 1) % len(domain_names)])
        for i in range(len(domain_names))
    ]


def _parse_transfer_pairs(
    pair_specs: Sequence[str] | None, domain_names: Sequence[str]
) -> List[Tuple[str, str]]:
    if not pair_specs:
        return _default_transfer_pairs(domain_names)

    pairs: List[Tuple[str, str]] = []
    known_domains = set(domain_names)
    for spec in pair_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --transfer-pair value '{spec}'. Use the form source:target."
            )

        source, target = (part.strip() for part in spec.split(":", 1))
        if source == target:
            raise ValueError(
                f"Invalid --transfer-pair value '{spec}': source and target must differ."
            )
        if source not in known_domains or target not in known_domains:
            raise ValueError(
                f"Invalid --transfer-pair value '{spec}': both domains must be declared with --domain."
            )

        pairs.append((source, target))

    return pairs


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Active Reference Database Adaptation experiment runner."
    )
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeatable domain definition, e.g. --domain kga=/data/kga.h5",
    )
    parser.add_argument(
        "--transfer-pair",
        action="append",
        default=None,
        metavar="SOURCE:TARGET",
        help=(
            "Repeatable directed transfer scenario. "
            "Defaults to a cyclic chain across the provided domain order."
        ),
    )
    parser.add_argument(
        "--taxonomy-map",
        type=Path,
        default=None,
        help="Optional CSV mapping dataset-specific labels to a shared taxonomy.",
    )
    parser.add_argument(
        "--taxonomy-domain-key",
        action="append",
        default=None,
        metavar="DOMAIN=DATASET_NAME",
        help=(
            "Optional mapping from local domain name to the dataset name used in the "
            "taxonomy CSV, e.g. --taxonomy-domain-key kga=\"KGA\""
        ),
    )
    parser.add_argument(
        "--taxonomy-dataset-column",
        type=str,
        default=None,
        help="Optional explicit taxonomy CSV column for dataset name matching.",
    )
    parser.add_argument(
        "--taxonomy-label-column",
        type=str,
        default=None,
        help="Optional explicit taxonomy CSV column for original label matching.",
    )
    parser.add_argument(
        "--taxonomy-target-column",
        type=str,
        default=None,
        help=(
            "Optional explicit taxonomy CSV column to use as the harmonized class label. "
            "Defaults to scientific_name when available."
        ),
    )
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--k-factor", type=float, default=0.5)
    parser.add_argument("--init-train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not 0.0 < args.init_train_ratio < 1.0:
        raise ValueError("--init-train-ratio must be strictly between 0 and 1.")

    domain_paths = _parse_domain_specs(args.domain)
    taxonomy_domain_keys = _parse_key_value_specs(
        args.taxonomy_domain_key, "--taxonomy-domain-key"
    )
    transfer_pairs = _parse_transfer_pairs(args.transfer_pair, list(domain_paths.keys()))

    return ExperimentConfig(
        domain_paths=domain_paths,
        transfer_pairs=transfer_pairs,
        taxonomy_mapping_path=args.taxonomy_map,
        taxonomy_domain_keys=taxonomy_domain_keys,
        taxonomy_dataset_column=args.taxonomy_dataset_column,
        taxonomy_label_column=args.taxonomy_label_column,
        taxonomy_target_column=args.taxonomy_target_column,
        n_blocks=args.n_blocks,
        k_factor=args.k_factor,
        init_train_ratio=args.init_train_ratio,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    run_experiment(config)
