import os
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import h5py
from PIL import Image
import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


# -- Paths -------------------------------------------------------------------
dataset = 'cdb'
METADATA_JSON = f"{dataset}_images.json"
MD_JSON = f"{dataset}_md.json"
DATA_DIR = f"../../../media/Data-10T-1/Bhavesh-project/{dataset}_data"
OUTPUT_H5 = f"../embeddings/{dataset}_dinov2l_embeddings_v2.h5"

# -- Config ------------------------------------------------------------------
DINOV2_MODEL = "facebook/dinov2-large"
PADDING_FRACTION = 0.10
BATCH_SIZE = 32
EMPTY_LABELS = {"empty", "human"}
ANIMAL_CATEGORY = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


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


def crop_with_padding(
    image: Image.Image, bbox: list, pad: float = PADDING_FRACTION
) -> Image.Image:
    """
    Crop a square region around a bounding box with padding on all sides.
    Produces a square output to avoid aspect ratio distortion when resized by DINOv2.
    If the detection is near an edge, the square shifts inward rather than going out of bounds.

    Args:
        image: PIL Image
        bbox: MegaDetector normalized bbox [x_min, y_min, width, height] in range [0, 1]
        pad: fractional padding relative to bbox dimensions
    """
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

    return {
        "category": ANIMAL_CATEGORY,
        "conf": conf,
        "bbox": bbox,
    }


def build_md_lookup(md_json_path):
    print("Loading MegaDetector JSON...")
    with open(md_json_path) as f:
        md = json.load(f)

    lookup = {}
    invalid_animal_detections = 0
    for image in md["images"]:
        animal_dets = []
        for det in image.get("detections", []):
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

    stats = {
        "images_with_valid_animal_detection": len(lookup),
        "invalid_animal_detections": invalid_animal_detections,
    }
    return lookup, stats


def build_image_records(metadata_json_path):
    print("Loading metadata...")
    with open(metadata_json_path) as f:
        data = json.load(f)

    category_name_by_id = {
        category["id"]: category["name"].lower().strip()
        for category in data["categories"]
    }
    images_by_id = {image["id"]: image for image in data["images"]}
    valid_annotation_count_by_image = Counter()
    for annotation in data["annotations"]:
        species = category_name_by_id.get(annotation["category_id"], "")
        if species and species not in EMPTY_LABELS:
            valid_annotation_count_by_image[annotation["image_id"]] += 1

    multi_annotated_ids = {
        image_id
        for image_id, count in valid_annotation_count_by_image.items()
        if count > 1
    }

    records = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id in multi_annotated_ids:
            continue

        species = category_name_by_id.get(annotation["category_id"], "")
        if not species or species in EMPTY_LABELS:
            continue

        image = images_by_id.get(image_id)
        if image is None:
            continue

        local_name = dataset_path_to_local_name(image["file_name"])
        if local_name in records and records[local_name]["image_id"] != image_id:
            raise ValueError(
                f"Collision after filename canonicalization: {local_name} maps to "
                f"{records[local_name]['image_id']} and {image_id}"
            )

        records[local_name] = {
            "image_id": image_id,
            "file_name": image["file_name"],
            "species": species,
            "location": image.get("location") or annotation.get("location") or "unknown",
            "date_captured": image.get("date_captured")
            or image.get("datetime")
            or annotation.get("datetime")
            or "unknown",
        }

    stats = {
        "total_annotations": len(data["annotations"]),
        "unique_images": len(images_by_id),
        "multi_annotated_images": len(multi_annotated_ids),
        "usable_images": len(records),
    }
    return records, stats


md_lookup, md_stats = build_md_lookup(MD_JSON)
image_records, metadata_stats = build_image_records(METADATA_JSON)

output_h5_path = Path(OUTPUT_H5)
output_h5_path.parent.mkdir(parents=True, exist_ok=True)

data_dir = Path(DATA_DIR)
local_files = sorted(list(data_dir.glob("*.JPG")) + list(data_dir.glob("*.jpg")))
work = []
skipped_no_metadata = 0
skipped_no_det = 0

for local_path in tqdm(local_files, desc="Building work list"):
    record = image_records.get(local_path.name)
    if record is None:
        skipped_no_metadata += 1
        continue

    det = md_lookup.get(local_path.name)
    if det is None:
        skipped_no_det += 1
        continue

    work.append((local_path, record, det))

print(
    "\nTo embed: {to_embed} | No metadata/species: {no_metadata} | "
    "No MD detection: {no_det} | Multi-annotation images skipped: {multi} | "
    "Invalid MD animal detections ignored: {invalid_md}".format(
        to_embed=len(work),
        no_metadata=skipped_no_metadata,
        no_det=skipped_no_det,
        multi=metadata_stats["multi_annotated_images"],
        invalid_md=md_stats["invalid_animal_detections"],
    )
)

print(f"Loading {DINOV2_MODEL}...")
processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
model = AutoModel.from_pretrained(DINOV2_MODEL).to(DEVICE).eval()

already_done = set()
if output_h5_path.exists():
    with h5py.File(output_h5_path, "r") as hf:
        if "file_name" in hf:
            already_done = {decode_h5_string(name) for name in hf["file_name"][:]}
    print(f"Resuming: {len(already_done)} embeddings already exist.")

eligible_work_items = sum(
    1 for _, record, _ in work if record["file_name"] not in already_done
)

with h5py.File(output_h5_path, "a") as hf:
    for ds_name, shape, dtype in [
        ("embeddings", (0, 1024), "float32"),
        ("file_name", (0,), h5py.string_dtype()),
        ("species", (0,), h5py.string_dtype()),
        ("location", (0,), h5py.string_dtype()),
        ("date_captured", (0,), h5py.string_dtype()),
    ]:
        if ds_name not in hf:
            max_shape = (None, 1024) if ds_name == "embeddings" else (None,)
            chunks = (64, 1024) if ds_name == "embeddings" else True
            hf.create_dataset(
                ds_name,
                shape=shape,
                maxshape=max_shape,
                dtype=dtype,
                chunks=chunks,
            )

batch_crops = []
batch_metadata = []
embedded_count = 0
error_count = 0
resumed_count = 0


def flush_batch(hf, crops, meta):
    global embedded_count
    inputs = processor(images=crops, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

    n = len(hf["embeddings"])
    m = len(crops)
    for key in ["embeddings", "file_name", "species", "location", "date_captured"]:
        hf[key].resize(n + m, axis=0)

    hf["embeddings"][n : n + m] = cls_embeddings
    hf["file_name"][n : n + m] = [item["file_name"].encode() for item in meta]
    hf["species"][n : n + m] = [item["species"].encode() for item in meta]
    hf["location"][n : n + m] = [item["location"].encode() for item in meta]
    hf["date_captured"][n : n + m] = [item["date_captured"].encode() for item in meta]
    embedded_count += m
    crops.clear()
    meta.clear()


with h5py.File(output_h5_path, "a") as hf:
    for local_path, record, det in tqdm(work, desc="Embedding"):
        if record["file_name"] in already_done:
            resumed_count += 1
            continue

        try:
            with Image.open(local_path) as image:
                crop = crop_with_padding(image.convert("RGB"), det["bbox"])
            batch_crops.append(crop)
            batch_metadata.append(
                {
                    "file_name": record["file_name"],
                    "species": record["species"],
                    "location": record["location"],
                    "date_captured": record["date_captured"],
                }
            )
        except Exception as exc:
            error_count += 1
            print(f"Error processing {local_path}: {exc}")
            continue

        if len(batch_crops) >= BATCH_SIZE:
            flush_batch(hf, batch_crops, batch_metadata)

    if batch_crops:
        flush_batch(hf, batch_crops, batch_metadata)

print("Processing complete.")
print(
    "Final stats | Total local images: {total_local} | Worklist matches: {worklist} | "
    "Already embedded: {resumed} | Newly embedded: {embedded} | "
    "Errors during processing: {errors}".format(
        total_local=len(local_files),
        worklist=len(work),
        resumed=resumed_count,
        embedded=embedded_count,
        errors=error_count,
    )
)
print(
    "Ignored images | No metadata/species: {no_metadata} | No MD detection: {no_det} | "
    "Multi-annotation images skipped: {multi} | Invalid MD animal detections ignored: {invalid_md}".format(
        no_metadata=skipped_no_metadata,
        no_det=skipped_no_det,
        multi=metadata_stats["multi_annotated_images"],
        invalid_md=md_stats["invalid_animal_detections"],
    )
)
print(
    "Eligibility check | Ready for embedding this run: {eligible} | "
    "Processed this run: {processed}".format(
        eligible=eligible_work_items,
        processed=embedded_count + error_count,
    )
)
