from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

#current folder contain all images
SOURCE_IMAGES = DATASET_ROOT / "train" / "images"
SOURCE_LABELS = DATASET_ROOT / "train" / "labels"

#temporary output location
OUTPUT_ROOT = DATASET_ROOT / "split_dataset"

TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)


#find images
image_extentions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

images = [
    image
    for image in SOURCE_IMAGES.iterdir()
    if image.suffix.lower() in image_extentions
]

random.shuffle(images)

total = len(images)

train_count = int(total * TRAIN_RATIO)
validation_count = int(total * VALIDATION_RATIO)

train_images = images[:train_count]

validation_images = images[
    train_count:
    train_count + validation_count:
]

test_images = images[
    train_count + validation_count:
]

print(f"Total images: {total}")
print(f"Train: {len(train_images)}")
print(f"Valid: {len(validation_images)}")
print(f"Test: {len(test_images)}")


# -------------------------------------------------
# Create directories
# -------------------------------------------------

for split in ["train", "valid", "test"]:

    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# -------------------------------------------------
# Copy image and matching label
# -------------------------------------------------

def copy_files(image_list, split_name):

    for image_path in image_list:

        label_path = (
            SOURCE_LABELS /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"Warning: Missing label for "
                f"{image_path.name}"
            )

            continue


        shutil.copy2(
            image_path,
            OUTPUT_ROOT /
            split_name /
            "images" /
            image_path.name
        )


        shutil.copy2(
            label_path,
            OUTPUT_ROOT /
            split_name /
            "labels" /
            label_path.name
        )


copy_files(
    train_images,
    "train"
)

copy_files(
    validation_images,
    "valid"
)

copy_files(
    test_images,
    "test"
)


print("\nDataset split completed.")
print(f"Saved to: {OUTPUT_ROOT}")