# src/meter_reader/split_data.py
import random
import shutil
from pathlib import Path


def split_dataset(source_dir: Path, target_dir: Path, split_ratio: float = 0.8):
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Dataset folders not found under {source_dir}. "
            f"Expected: {images_dir} and {labels_dir}"
        )

    raw_images = list(images_dir.glob("*.*"))
    if not raw_images:
        raise FileNotFoundError(f"No image files were found in {images_dir}")

    random.seed(42)
    random.shuffle(raw_images)

    split_idx = int(len(raw_images) * split_ratio)
    train_images = raw_images[:split_idx]
    val_images = raw_images[split_idx:]

    for split_name, img_list in [("train", train_images), ("val", val_images)]:
        img_dest = target_dir / "images" / split_name
        lbl_dest = target_dir / "labels" / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            shutil.copy(img_path, img_dest / img_path.name)

            lbl_file = labels_dir / f"{img_path.stem}.txt"
            if lbl_file.exists():
                shutil.copy(lbl_file, lbl_dest / lbl_file.name)
            else:
                (lbl_dest / f"{img_path.stem}.txt").touch()

    print(f"Dataset split complete: {len(train_images)} train, {len(val_images)} val.")
    print(f"Output written to: {target_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    split_dataset(
        source_dir=project_root / "dataset",
        target_dir=project_root / "dataset" / "processed",
    )