#!/usr/bin/env bash

set -euo pipefail

# Update these paths to your actual H5 files before running.
KGA_H5="/Projects/Bhavesh_project/embeddings/kga_dinov2l_embeddings_v2.h5"
KAR_H5="/Projects/Bhavesh_project/embeddings/kar_dinov2l_embeddings_v2.h5"
KRU_H5="/Projects/Bhavesh_project/embeddings/kru_dinov2l_embeddings_v2.h5"
#CDB_H5="/Projects/Bhavesh_project/embeddings/cdb_dinov2l_embeddings_v2.h5"
ENO_H5="/Projects/Bhavesh_project/embeddings/eno_dinov2l_embeddings_v2.h5"
TAXONOMY_CSV="/Projects/Bhavesh_project/cct_notebooks/json_files/lila-taxonomy-mapping.csv"

# If the mapping CSV uses different dataset names than kga/kar/kru/cdb/eno,
# set them here. Otherwise leave them equal to the local domain names.
KGA_TAXONOMY_KEY="Snapshot Kgalagadi"
KAR_TAXONOMY_KEY="Snapshot Karoo"
KRU_TAXONOMY_KEY="Snapshot Kruger"
ENO_TAXONOMY_KEY="Snapshot Enonkishu"

python3 run_experiment.py \
  --domain "kga=${KGA_H5}" \
  --domain "kar=${KAR_H5}" \
  --domain "kru=${KRU_H5}" \
  --domain "eno=${ENO_H5}" \
  --taxonomy-map "${TAXONOMY_CSV}" \
  --taxonomy-domain-key "kga=${KGA_TAXONOMY_KEY}" \
  --taxonomy-domain-key "kar=${KAR_TAXONOMY_KEY}" \
  --taxonomy-domain-key "kru=${KRU_TAXONOMY_KEY}" \
  --taxonomy-domain-key "eno=${ENO_TAXONOMY_KEY}" \
  --taxonomy-label-column query \
  --taxonomy-target-column scientific_name \
  --transfer-pair kga:kar \
  --transfer-pair kar:kru \
  --transfer-pair kru:eno \
  --transfer-pair kar:eno \
  --n-blocks 5 \
  --k-factor 0.5 \
  --init-train-ratio 0.8 \
  --output-dir outputs
