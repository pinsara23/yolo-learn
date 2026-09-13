from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "split_dataset"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

def check_split(split_name: str):
    image_dir = DATASET_ROOT / split_name / "images"
    label_dir = DATASET_ROOT / split_name / "labels"

    print(f"Checking {split_name} dataset...")
    print("-" * 40)

    if not image_dir.exists():
        print(f"Image directory does not exist: {image_dir}")
        return

    if not label_dir.exists():
        print(f"Label directory does not exist: {label_dir}")
        return

    images = [
        file 
        for file in image_dir.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")

    missing_labels = []

    for image in images:
        label_file = label_dir / f"{image.stem}.txt"
        if not label_file.exists():
            missing_labels.append(image.name)

    if missing_labels:
        print("\nMissing labels for the following images:")

        for name in missing_labels:
            print(name)
    else:
        print("\nAll images have corresponding labels.")

check_split("train")
check_split("test")
check_split("valid")