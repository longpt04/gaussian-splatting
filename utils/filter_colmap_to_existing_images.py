import argparse
import os
import shutil
from pathlib import Path

from read_write_model import read_model, write_model


def main():
    parser = argparse.ArgumentParser(
        description="Keep only COLMAP image entries whose image files exist."
    )
    parser.add_argument("--scene", required=True, help="Scene root containing images/ and sparse/0/")
    parser.add_argument("--images", default="images", help="Image folder relative to scene root")
    parser.add_argument("--sparse", default="sparse/0", help="Sparse model folder relative to scene root")
    parser.add_argument(
        "--keep-points3d",
        action="store_true",
        help="Keep points3D entries unchanged and only filter images/cameras.",
    )
    args = parser.parse_args()

    scene = Path(args.scene)
    images_dir = scene / args.images
    sparse_dir = scene / args.sparse

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

    existing_names = {p.name for p in images_dir.iterdir() if p.is_file()}
    cameras, images, points3D = read_model(str(sparse_dir), ".bin")

    kept_images = {
        image_id: image for image_id, image in images.items() if image.name in existing_names
    }
    kept_image_ids = set(kept_images.keys())
    kept_camera_ids = {image.camera_id for image in kept_images.values()}
    kept_cameras = {
        camera_id: camera for camera_id, camera in cameras.items() if camera_id in kept_camera_ids
    }

    if args.keep_points3d:
        kept_points3D = points3D
    else:
        kept_points3D = {}
        for point_id, point in points3D.items():
            keep_track = [idx for idx, image_id in enumerate(point.image_ids) if image_id in kept_image_ids]
            if not keep_track:
                continue
            kept_points3D[point_id] = point._replace(
                image_ids=point.image_ids[keep_track],
                point2D_idxs=point.point2D_idxs[keep_track],
            )

    backup_dir = sparse_dir.with_name(sparse_dir.name + "_unfiltered")
    if not backup_dir.exists():
        shutil.copytree(sparse_dir, backup_dir)

    write_model(kept_cameras, kept_images, kept_points3D, str(sparse_dir), ".bin")

    missing = sorted(image.name for image in images.values() if image.name not in existing_names)
    print(
        "Filtered COLMAP model: "
        f"images {len(images)} -> {len(kept_images)}, "
        f"points3D {len(points3D)} -> {len(kept_points3D)}"
    )
    if args.keep_points3d:
        print("Kept points3D unchanged; only image/camera entries were filtered.")
    if missing:
        print(f"Missing image files ignored: {len(missing)}")
        print("First missing images: " + ", ".join(missing[:10]))


if __name__ == "__main__":
    main()
