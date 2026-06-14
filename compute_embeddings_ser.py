import os
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

# ─── PATHS & DATA CONFIG ──────────────────────────────────────────────────────
DATASET = 'SnapshotSerengetiS01'
METADATA_JSON = f"./json_files/{DATASET}.json"
MD_JSON = f"./json_files/{DATASET}_md.json"  # Ensure this contains your MegaDetector output mapping
DATA_DIR = "../../../media/Data-10T-1/Bhavesh-project/ser_data"
OUTPUT_H5 = f"../embeddings/{DATASET}_dinov2l_bbox_embeddings.h5"

# ─── PROCESSING CONFIGURATION ─────────────────────────────────────────────────
DINOV2_MODEL = "facebook/dinov2-large"
PADDING_FRACTION = 0.10
BATCH_SIZE = 32
EMPTY_LABELS = {"empty", "human", "blank"}
ANIMAL_CATEGORY = "1"  # MegaDetector's default class ID for animals
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEHAVIOR_COLS = ["standing", "resting", "moving", "interacting", "young_present"]

print(f"Using device target: {DEVICE}")

def dataset_path_to_local_name(path_like: str) -> str:
    normalized = str(path_like).replace("\\", "/")
    return str(PurePosixPath(normalized)).replace("/", "_")

def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)

def choose_square_start(center: float, side: int, limit: int) -> int:
    if side >= limit:
        return 0
    start = int(round(center - side / 2))
    return min(max(start, 0), limit - side)

def crop_with_padding(image: Image.Image, bbox: list, pad: float = PADDING_FRACTION) -> Image.Image:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got {bbox}")

    bbox = [float(v) for v in bbox]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"Expected positive bbox width/height, got {bbox}")

    img_w, img_h = image.size
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {(img_w, img_h)}")

    x_min = min(max(bbox[0], 0.0), 1.0)
    y_min = min(max(bbox[1], 0.0), 1.0)
    x_max = min(max(x_min + bbox[2], x_min), 1.0)
    y_max = min(max(y_min + bbox[3], y_min), 1.0)

    box_w = (x_max - x_min) * img_w
    box_h = (y_max - y_min) * img_h
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Bbox collapses outside image bounds: {bbox}")

    x_min *= img_w
    y_min *= img_h

    pad_x = pad * box_w
    pad_y = pad * box_h

    padded_w = box_w + 2 * pad_x
    padded_h = box_h + 2 * pad_y
    side = min(max(padded_w, padded_h), float(min(img_w, img_h)))
    side_px = max(1, min(int(round(side)), img_w, img_h))

    cx = x_min + box_w / 2
    cy = y_min + box_h / 2

    x0 = choose_square_start(cx, side_px, img_w)
    y0 = choose_square_start(cy, side_px, img_h)
    x1 = x0 + side_px
    y1 = y0 + side_px

    return image.crop((x0, y0, x1, y1))

def normalize_detection(det):
    if str(det.get("category")) != ANIMAL_CATEGORY:
        return None
    bbox = det.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        conf = float(det["conf"])
        bbox = [float(value) for value in bbox]
    except (KeyError, TypeError, ValueError):
        return None
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return {"category": ANIMAL_CATEGORY, "conf": conf, "bbox": bbox}

def build_md_lookup(md_json_path):
    print("Loading MegaDetector Bounding Box Output Map JSON...")
    with open(md_json_path) as f:
        md = json.load(f)

    lookup = {}
    invalid_animal_detections = 0
    for image in md["images"]:
        animal_dets = []
        # Fallback to an empty list if "detections" is missing OR explicitly set to None
        detections = image.get("detections") or []
        
        for det in detections:
            normalized = normalize_detection(det)
            normalized = normalize_detection(det)
            if normalized is not None:
                animal_dets.append(normalized)
            elif str(det.get("category")) == ANIMAL_CATEGORY:
                invalid_animal_detections += 1

        if not animal_dets:
            continue

        local_name = dataset_path_to_local_name(image["file"])
        best_det = max(animal_dets, key=lambda det: det["conf"])
        existing = lookup.get(local_name)
        if existing is None or best_det["conf"] > existing["conf"]:
            lookup[local_name] = best_det

    return lookup, {"images_with_valid_animal_detection": len(lookup), "invalid_animal_detections": invalid_animal_detections}

def build_image_records(metadata_json_path):
    print("Parsing Dataset Source Annotations...")
    with open(metadata_json_path) as f:
        data = json.load(f)

    category_name_by_id = {c["id"]: c["name"].lower().strip() for c in data["categories"]}
    images_by_id = {image["id"]: image for image in data["images"]}
    
    valid_annotation_count_by_image = Counter()
    for annotation in data["annotations"]:
        species = category_name_by_id.get(annotation["category_id"], "")
        if species and species not in EMPTY_LABELS:
            valid_annotation_count_by_image[annotation["image_id"]] += 1

    multi_annotated_ids = {img_id for img_id, count in valid_annotation_count_by_image.items() if count > 1}

    records = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id in multi_annotated_ids:
            continue

        species = category_name_by_id.get(annotation["category_id"], "")
        if not species or species in EMPTY_LABELS:
            continue
            
        # Isolate instances containing incomplete or NaN behavioral features
        if any(pd.isna(annotation.get(col)) for col in BEHAVIOR_COLS):
            continue

        image = images_by_id.get(image_id)
        if image is None:
            continue

        local_name = dataset_path_to_local_name(image["file_name"])
        if local_name in records and records[local_name]["image_id"] != image_id:
            raise ValueError(f"Collision encountered on path name: {local_name}")

        # Storing image metadata and behavioral coordinates
        records[local_name] = {
            "image_id": image_id,
            "file_name": image["file_name"],
            "species": species,
            "behaviors": {col: float(annotation.get(col, 0.0)) for col in BEHAVIOR_COLS}
        }

    return records, {"total_annotations": len(data["annotations"]), "multi_annotated_images": len(multi_annotated_ids), "usable_images": len(records)}

# ─── CORE PIPELINE INITIALIZATION ─────────────────────────────────────────────
md_lookup, md_stats = build_md_lookup(MD_JSON)
image_records, metadata_stats = build_image_records(METADATA_JSON)

output_h5_path = Path(OUTPUT_H5)
output_h5_path.parent.mkdir(parents=True, exist_ok=True)

data_dir = Path(DATA_DIR)
local_files = sorted(list(data_dir.glob("*.JPG")) + list(data_dir.glob("*.jpg")))
total_local_files = len(local_files)

work = []
skipped_no_metadata = 0
skipped_no_det = 0

for local_path in tqdm(local_files, desc="Syncing Work Distribution"):
    record = image_records.get(local_path.name)
    if record is None:
        skipped_no_metadata += 1
        continue

    det = md_lookup.get(local_path.name)
    if det is None:
        skipped_no_det += 1
        continue

    work.append((local_path, record, det))

print(f"\nProcessing Task Allocation -> Target Processing Volume: {len(work)}")

# ─── INITIALIZE VISION TRANSFORMER MODEL ──────────────────────────────────────
print(f"Loading feature extractor: {DINOV2_MODEL}...")
processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
model = AutoModel.from_pretrained(DINOV2_MODEL).to(DEVICE).eval()

already_done_count = 0
if output_h5_path.exists():
    with h5py.File(output_h5_path, "r") as hf:
        if "file_name" in hf:
            already_done = {decode_h5_string(name) for name in hf["file_name"][:]}
            already_done_count = sum(1 for _, rec, _ in work if rec["file_name"] in already_done)
    print(f"Resuming pipeline checklist: Found {len(already_done)} compiled entries.")
    
# ─── PIPELINE DROP-OFF STATISTICS REPORT ──────────────────────────────────────
print("\n" + "="*60)
print("          HIGH-LEVEL DATA PIPELINE STATISTICS          ")
print("="*60)
print(f"Total images found in local folder:           {total_local_files}")
print("-"*60)
print(f"FILTER 1: Dropped (Not in valid Metadata): {skipped_no_metadata} ({skipped_no_metadata/total_local_files:.1%})")
print(f"   ↳ Remaining after Metadata Filter:          {total_local_files - skipped_no_metadata}")
print("-"*60)
print(f"FILTER 2: Dropped (No MegaDetector Animal):{skipped_no_det} ({skipped_no_det/total_local_files:.1%})")
print(f"   ↳ Remaining after MegaDetector Filter:      {len(work)}")
print("-"*60)
print(f"SKIP: Already processed in HDF5 file:      {already_done_count}")
print(f"TARGET: Net new images to embed this run:   {len(work) - already_done_count}")
print("="*60 + "\n")

# Quick context on why metadata drops happened from your metadata parser:
print(f"Context from Metadata JSON parsing:")
print(f"  • Total base annotations evaluated:          {metadata_stats['total_annotations']}")
print(f"  • Skipped due to multi-species clusters:     {metadata_stats['multi_annotated_images']}")
print(f"  • Total clean/usable metadata records:       {metadata_stats['usable_images']}\n")

# ─── CONSTRUCT HDF5 STORE SCHEMAS ─────────────────────────────────────────────
with h5py.File(output_h5_path, "a") as hf:
    datasets_config = [
        ("embeddings", (0, 1024), "float32", (None, 1024), (64, 1024)),
        ("behaviors", (0, 5), "int32", (None, 5), (64, 5)),  # Shape maps to 5 target behaviors
        ("file_name", (0,), h5py.string_dtype(), (None,), True),
        ("image_id", (0,), h5py.string_dtype(), (None,), True),
        ("species", (0,), h5py.string_dtype(), (None,), True),
    ]
    for ds_name, shape, dtype, max_shape, chunks in datasets_config:
        if ds_name not in hf:
            hf.create_dataset(ds_name, shape=shape, maxshape=max_shape, dtype=dtype, chunks=chunks)

batch_crops = []
batch_metadata = []
embedded_count = 0

def flush_batch(hf, crops, meta):
    global embedded_count
    inputs = processor(images=crops, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

    # Convert continuous scores to hard binary integers using the >0.5 cutoff
    binary_behaviors = np.array([
        [1 if item["behaviors"][col] > 0.5 else 0 for col in BEHAVIOR_COLS]
        for item in meta
    ], dtype=np.int32)

    n = len(hf["embeddings"])
    m = len(crops)
    
    for key in ["embeddings", "behaviors", "file_name", "image_id", "species"]:
        hf[key].resize(n + m, axis=0)

    hf["embeddings"][n : n + m] = cls_embeddings
    hf["behaviors"][n : n + m] = binary_behaviors
    hf["file_name"][n : n + m] = [item["file_name"].encode() for item in meta]
    hf["image_id"][n : n + m] = [item["image_id"].encode() for item in meta]
    hf["species"][n : n + m] = [item["species"].encode() for item in meta]
    
    embedded_count += m
    crops.clear()
    meta.clear()

# ─── EXTRACT AND STREAM TO HDF5 ───────────────────────────────────────────────
with h5py.File(output_h5_path, "a") as hf:
    for local_path, record, det in tqdm(work, desc="Extracting BBox Patches & Embeddings"):
        if record["file_name"] in already_done:
            continue

        try:
            with Image.open(local_path) as image:
                crop = crop_with_padding(image.convert("RGB"), det["bbox"])
            batch_crops.append(crop)
            batch_metadata.append(record)
        except Exception as exc:
            print(f"Skipping corrupt asset {local_path}: {exc}")
            continue

        if len(batch_crops) >= BATCH_SIZE:
            flush_batch(hf, batch_crops, batch_metadata)

    if batch_crops:
        flush_batch(hf, batch_crops, batch_metadata)

print(f"\nPipeline run successful. Newly recorded patch instances: {embedded_count}")