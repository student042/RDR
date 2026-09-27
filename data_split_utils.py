import json
import os
import random
from collections import Counter, defaultdict


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object mapping sample ids to annotations")
    return data


def _write_json(path, data):
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def prepare_stratified_train_val_split(
    source_train_json,
    output_dir,
    val_ratio=0.1,
    seed=42,
    stratified=True,
):
    """Create a deterministic train/validation split without changing source data."""
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    source_train_json = os.path.abspath(source_train_json)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    annotations = _load_json(source_train_json)
    rng = random.Random(int(seed))
    groups = defaultdict(list)
    if stratified:
        for sample_id, record in annotations.items():
            groups[int(record["label"])].append(sample_id)
    else:
        groups["all"] = list(annotations)

    train_ids = []
    val_ids = []
    for group_name in sorted(groups, key=str):
        group_ids = sorted(groups[group_name], key=str)
        rng.shuffle(group_ids)
        val_count = int(round(len(group_ids) * float(val_ratio)))
        if len(group_ids) > 1:
            val_count = min(max(val_count, 1), len(group_ids) - 1)
        else:
            val_count = 0
        val_ids.extend(group_ids[:val_count])
        train_ids.extend(group_ids[val_count:])

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    if train_id_set & val_id_set:
        raise RuntimeError("Train and validation ids overlap")
    if train_id_set | val_id_set != set(annotations):
        raise RuntimeError("Split does not cover the source annotations")

    # Preserve the source file's deterministic sample order.
    train_data = {
        sample_id: record
        for sample_id, record in annotations.items()
        if sample_id in train_id_set
    }
    val_data = {
        sample_id: record
        for sample_id, record in annotations.items()
        if sample_id in val_id_set
    }

    train_path = os.path.join(output_dir, "train_sub.json")
    val_path = os.path.join(output_dir, "val_sub.json")
    manifest_path = os.path.join(output_dir, "manifest.json")
    _write_json(train_path, train_data)
    _write_json(val_path, val_data)

    manifest = {
        "source_train_json": source_train_json,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "stratified": bool(stratified),
        "total_count": len(annotations),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "source_class_counts": dict(
            sorted(Counter(int(v["label"]) for v in annotations.values()).items())
        ),
        "train_class_counts": dict(
            sorted(Counter(int(v["label"]) for v in train_data.values()).items())
        ),
        "val_class_counts": dict(
            sorted(Counter(int(v["label"]) for v in val_data.values()).items())
        ),
        "train_ids": list(train_data),
        "val_ids": list(val_data),
        "train_json": train_path,
        "val_json": val_path,
    }
    _write_json(manifest_path, manifest)
    return train_path, val_path, manifest
