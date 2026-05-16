#!/usr/bin/env python3
import sys
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import logging
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import argparse

# =============================
# ARGUMENT PARSING
# =============================
parser = argparse.ArgumentParser(description="Convert subfolder-structured TIFF files to HDF5.")
parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--h5_name", required=True)
parser.add_argument("--manifest_name", required=True)
parser.add_argument("--parquet_name", required=True)
parser.add_argument("--log_name", required=True)
args = parser.parse_args()

root_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
h5_path = output_dir / args.h5_name
manifest_path = output_dir / args.manifest_name
parquet_path = output_dir / args.parquet_name
log_path = output_dir / args.log_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
)

def sanitize_name(path_obj, root_obj):
    rel = path_obj.relative_to(root_obj)
    return str(rel).replace("/", "_").replace("\\", "_").replace(".", "_")

# =============================
# PROCESSING LOGIC
# =============================
subfolders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
logging.info(f"Found {len(subfolders)} subfolders in {root_dir}")

parquet_schema = pa.schema([("image_id", pa.string()), ("hu_value", pa.int32())])

with h5py.File(h5_path, "w") as h5f, \
     open(manifest_path, "w") as manifest, \
     ParquetWriter(str(parquet_path), parquet_schema) as pq_writer:

    manifest.write(f"=== HDF5 MANIFEST: {args.h5_name} ===\n\n")
    
    img_group = h5f.create_group("images")
    hu_group = h5f.create_group("hu_values")

    for folder in subfolders:
        manifest.write(f"SUBFOLDER: {folder.name}\n")
        
        # Process TIFFs - Added filter for hidden metadata files
        tifs = sorted([f for f in list(folder.glob("*.tif")) + list(folder.glob("*.tiff")) 
                       if not f.name.startswith("._")])
        
        for tif_p in tifs:
            try:
                # This requires imagecodecs if inputs are LZW compressed
                img_data = tifffile.imread(tif_p)
                ds_name = sanitize_name(tif_p, root_dir)

                # Save using GZIP compression (Level 4)
                img_group.create_dataset(ds_name, data=img_data, compression="gzip", compression_opts=4)
                
                hu_flat = img_data.ravel().astype(np.int32())
                hu_group.create_dataset(ds_name, data=hu_flat, compression="gzip", compression_opts=4)

                hu_table = pa.Table.from_arrays(
                    [pa.array([ds_name] * len(hu_flat)), pa.array(hu_flat)],
                    schema=parquet_schema
                )
                pq_writer.write_table(hu_table)
                manifest.write(f"  [TIF] {tif_p.name} | Shape: {img_data.shape}\n")
                
                del img_data 
                del hu_flat
            except Exception as e:
                logging.error(f"Failed TIF {tif_p}: {e}")

        manifest.write("\n")

logging.info(f"Process complete.")