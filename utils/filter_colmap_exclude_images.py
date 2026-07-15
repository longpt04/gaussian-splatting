import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from read_write_model import read_model, write_model


def read_excluded_names(path: Path, column: str):
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if column not in reader.fieldnames:
                raise ValueError(f"Column '{column}' not found in {path}; columns={reader.fieldnames}")
            return {row[column] for row in reader if row.get(column)}

    with path.open() as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def main():
    parser = argparse.ArgumentParser(
        description="Copy/filter a COLMAP binary model, excluding images by filename."
    )
    parser.add_argument("--input_sparse", required=True, help="Input sparse/0 folder")
    parser.add_argument("--output_sparse", required=True, help="Output sparse/0 folder")
    parser.add_argument("--exclude_names", required=True, help="CSV or text file with image names to exclude")
    parser.add_argument("--column", default="image_name", help="CSV column containing image names")
    args = parser.parse_args()

    input_sparse = Path(args.input_sparse)
    output_sparse = Path(args.output_sparse)
    exclude_names = read_excluded_names(Path(args.exclude_names), args.column)

    if not input_sparse.is_dir():
        raise FileNotFoundError(f"Input sparse folder not found: {input_sparse}")
    output_sparse.parent.mkdir(parents=True, exist_ok=True)
    if output_sparse.exists():
        shutil.rmtree(output_sparse)
    output_sparse.mkdir(parents=True)

    cameras, images, points3D = read_model(str(input_sparse), ".bin")

    kept_images = {
        image_id: image for image_id, image in images.items() if image.name not in exclude_names
    }
    kept_image_ids = set(kept_images.keys())
    kept_camera_ids = {image.camera_id for image in kept_images.values()}
    kept_cameras = {
        camera_id: camera for camera_id, camera in cameras.items() if camera_id in kept_camera_ids
    }

    kept_points3D = {}
    kept_point_ids = set()
    for point_id, point in points3D.items():
        keep_track = np.array(
            [idx for idx, image_id in enumerate(point.image_ids) if image_id in kept_image_ids],
            dtype=np.int64,
        )
        if len(keep_track) == 0:
            continue
        kept_points3D[point_id] = point._replace(
            image_ids=point.image_ids[keep_track],
            point2D_idxs=point.point2D_idxs[keep_track],
        )
        kept_point_ids.add(point_id)

    # Keep each image's 2D observations, but clear references to removed 3D points.
    cleaned_images = {}
    for image_id, image in kept_images.items():
        point3D_ids = image.point3D_ids.copy()
        invalid = np.array([pid not in kept_point_ids for pid in point3D_ids], dtype=bool)
        point3D_ids[invalid] = -1
        cleaned_images[image_id] = image._replace(point3D_ids=point3D_ids)

    write_model(kept_cameras, cleaned_images, kept_points3D, str(output_sparse), ".bin")

    removed = sorted(image.name for image in images.values() if image.name in exclude_names)
    remaining_overlap = sorted(image.name for image in cleaned_images.values() if image.name in exclude_names)
    print(
        "Filtered COLMAP model by excluded names: "
        f"images {len(images)} -> {len(cleaned_images)}, "
        f"points3D {len(points3D)} -> {len(kept_points3D)}"
    )
    print(f"Excluded names requested : {len(exclude_names)}")
    print(f"Excluded names removed   : {len(removed)}")
    if removed:
        print("First removed images: " + ", ".join(removed[:10]))
    if remaining_overlap:
        raise RuntimeError("Excluded image names remain after filtering: " + ", ".join(remaining_overlap[:10]))


if __name__ == "__main__":
    main()
