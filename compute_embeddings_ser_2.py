import os
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from tqdm.auto import tqdm
from google.cloud import storage
from transformers import AutoImageProcessor, AutoModel

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATASET = 'SnapshotSerengetiS01'
METADATA_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-v-2-0/SnapshotSerengetiS01.json.zip"
ZIP_FILE = "snapshot_serengeti_s1.json.zip"
JSON_FILE = f"./json_files/{DATASET}.json"
MD_JSON = f"./json_files/{DATASET}_md.json"  # Must be present before running
DATA_DIR = "../../../media/Data-10T-1/Bhavesh-project/ser_data"
OUTPUT_H5 = f"../embeddings/{DATASET}_dinov2l_bbox_embeddings.h5"

# Processing Hyperparameters
DINOV2_MODEL = "facebook/dinov2-large"
PADDING_FRACTION = 0.10
BATCH_SIZE = 32
NUM_WORKERS = 8
SAMPLES_PER_SPECIES = 300
EMPTY_LABELS = {"empty", "human", "blank"}
ANIMAL_CATEGORY = "1"  # MegaDetector animal class ID
BEHAVIOR_COLS = ["standing", "resting", "moving", "interacting", "young_present"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
print(f"Using device target: {DEVICE}")

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────
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
    return image.crop((x0, y0, x0 + side_px, y0 + side_px))

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

# ─── STEP 1: PARSE MEGADETECTOR FIRST (THE WHITELIST) ─────────────────────────
print("Loading MegaDetector Bounding Box Output Map JSON...")
if not os.path.exists(MD_JSON):
    raise FileNotFoundError(f"Missing required MegaDetector mapping file at: {MD_JSON}")

with open(MD_JSON) as f:
    md = json.load(f)

md_lookup = {}
for image in md["images"]:
    animal_dets = []
    detections = image.get("detections") or []
    for det in detections:
        normalized = normalize_detection(det)
        if normalized is not None:
            animal_dets.append(normalized)

    if not animal_dets:
        continue

    local_name = dataset_path_to_local_name(image["file"])
    best_det = max(animal_dets, key=lambda det: det["conf"])
    
    # Store the best animal detection indexed by what its local filename will be
    md_lookup[local_name] = best_det

print(f"MegaDetector screening complete. Found {len(md_lookup)} images containing valid animal detections.")

# ─── STEP 2: DOWNLOAD & FILTER SERENGETI METADATA ────────────────────────────
if not os.path.exists(JSON_FILE):
    print("Downloading Snapshot Serengeti S1 metadata...")
    os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)
    urllib.request.urlretrieve(METADATA_URL, ZIP_FILE)
    print("Extracting metadata...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        z.extractall(os.path.dirname(JSON_FILE))
    os.remove(ZIP_FILE)
    print("Metadata extraction complete.")

print("Loading JSON metadata into memory...")
with open(JSON_FILE, "r") as f:
    metadata = json.load(f)

cat_id_to_name = {c["id"]: c["name"].lower().strip() for c in metadata["categories"]}
image_metadata_map = {img["id"]: img for img in metadata["images"]}
species_records = defaultdict(list)

print("Filtering annotations against MegaDetector and Behavior requirements...")
for ann in metadata["annotations"]:
    species_name = cat_id_to_name.get(ann["category_id"], "unknown")
    
    # Filter 1: Basic label cleanup
    if species_name in EMPTY_LABELS or species_name == "unknown":
        continue
        
    # Filter 2: Validate behavior metrics are non-NaN and present
    if any(pd.isna(ann.get(col)) for col in BEHAVIOR_COLS):
        continue
        
    # Get associated image entry
    img_entry = image_metadata_map.get(ann["image_id"])
    if not img_entry:
        continue
        
    # Filter 3: CRITICAL ADJUSTMENT — Verify image exists in MegaDetector's animal list
    local_name = dataset_path_to_local_name(img_entry["file_name"])
    if local_name not in md_lookup:
        continue  # Skip downloading this image entirely because MegaDetector didn't find an animal
        
    species_records[species_name].append(ann)

# ─── STEP 3: STRATIFIED SAMPLING ──────────────────────────────────────────────
print(f"\nPerforming stratified sampling ({SAMPLES_PER_SPECIES} images per class)...")
sampled_annotations = []
random.seed(42)

for species, records in species_records.items():
    if len(records) >= SAMPLES_PER_SPECIES:
        sampled = random.sample(records, SAMPLES_PER_SPECIES)
    else:
        sampled = records
        print(f"--> Warning: Species '{species}' only has {len(records)} records with valid bounding boxes. Taking all.")
    sampled_annotations.extend(sampled)

print(f"Total verified records selected for download: {len(sampled_annotations)}")

# ─── STEP 4: MULTI-THREADED GCS DOWNLOAD ──────────────────────────────────────
download_tasks = []
usable_records = {}

for ann in sampled_annotations:
    img_entry = image_metadata_map.get(ann["image_id"])
    file_name = img_entry["file_name"]
    local_name = dataset_path_to_local_name(file_name)
    local_path = os.path.join(DATA_DIR, local_name)
    
    # Store metadata schema map for processing down the line
    usable_records[local_name] = {
        "image_id": ann["image_id"],
        "file_name": file_name,
        "species": cat_id_to_name[ann["category_id"]],
        "behaviors": {col: float(ann.get(col, 0.0)) for col in BEHAVIOR_COLS}
    }
    
    if not os.path.exists(local_path):
        gcs_blob = "snapshotserengeti-unzipped/" + file_name
        download_tasks.append((gcs_blob, local_path))

if download_tasks:
    print(f"Images remaining to fetch (safely known to contain animals): {len(download_tasks)}")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("public-datasets-lila")

    def download_one(task):
        gcs_blob_path, local_path = task
        try:
            blob = bucket.blob(gcs_blob_path)
            blob.download_to_filename(local_path)
            return True, None
        except Exception as e:
            return False, str(e)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(download_one, t): t for t in download_tasks}
        with tqdm(total=len(download_tasks), desc="Downloading Filtered Assets") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
print("Download phase complete. Every local file now maps perfectly to metadata and a bounding box.")

# ─── STEP 5: VISION EMBEDDING GENERATION ──────────────────────────────────────
work = []
data_dir = Path(DATA_DIR)
local_files = sorted(list(data_dir.glob("*.JPG")) + list(data_dir.glob("*.jpg")))

for local_path in local_files:
    if local_path.name not in usable_records:
        continue
    det = md_lookup[local_path.name]  # Guaranteed to exist due to upfront validation
    work.append((local_path, usable_records[local_path.name], det))

print(f"\nProcessing Task Allocation -> Target Embedding Volume: {len(work)}")

print(f"Loading feature extractor: {DINOV2_MODEL}...")
processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
model = AutoModel.from_pretrained(DINOV2_MODEL).to(DEVICE).eval()

output_h5_path = Path(OUTPUT_H5)
already_done = set()
if output_h5_path.exists():
    with h5py.File(output_h5_path, "r") as hf:
        if "file_name" in hf:
            already_done = {decode_h5_string(name) for name in hf["file_name"][:]}

# Setup chunked HDF5 schemas
with h5py.File(output_h5_path, "a") as hf:
    datasets_config = [
        ("embeddings", (0, 1024), "float32", (None, 1024), (64, 1024)),
        ("behaviors", (0, 5), "int32", (None, 5), (64, 5)),
        ("file_name", (0,), h5py.string_dtype(), (None,), True),
        ("image_id", (0,), h5py.string_dtype(), (None,), True),
        ("species", (0,), h5py.string_dtype(), (None,), True),
    ]
    for ds_name, shape, dtype, max_shape, chunks in datasets_config:
        if ds_name not in hf:
            hf.create_dataset(ds_name, shape=shape, maxshape=max_shape, dtype=dtype, chunks=chunks)

def flush_batch(hf, crops, meta):
    inputs = processor(images=crops, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
    binary_behaviors = np.array([
        [1 if item["behaviors"][col] > 0.5 else 0 for col in BEHAVIOR_COLS] for item in meta
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
    crops.clear()
    meta.clear()

# Stream crops directly into HDF5 Storage
batch_crops, batch_metadata = [], []
embedded_count = 0

with h5py.File(output_h5_path, "a") as hf:
    for local_path, record, det in tqdm(work, desc="Extracting BBox Patches & Generating DINOv2 Embeddings"):
        if record["file_name"] in already_done:
            continue

        try:
            with Image.open(local_path) as image:
                crop = crop_with_padding(image.convert("RGB"), det["bbox"])
            batch_crops.append(crop)
            batch_metadata.append(record)
            embedded_count += 1
        except Exception as exc:
            print(f"Skipping corrupt asset {local_path}: {exc}")
            continue

        if len(batch_crops) >= BATCH_SIZE:
            flush_batch(hf, batch_crops, batch_metadata)

    if batch_crops:
        flush_batch(hf, batch_crops, batch_metadata)

print(f"\nPipeline finished completely! Total items stored into HDF5: {embedded_count}")