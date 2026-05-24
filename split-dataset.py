import os
import random
import shutil

SOURCE_DIR = "./dataset_sub_60k"
OUTPUT_ROOT = "./dataset-vit"
CLASSES = ["ai", "nature"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
RANDOM_SEED = 42

SPLIT_SIZES = {
    "train": 8000,
    "valid": 2000,
    "test": 1000,
}


def collect_images(class_dir):
    return [
        file_name
        for file_name in os.listdir(class_dir)
        if file_name.lower().endswith(IMAGE_EXTENSIONS)
    ]


def reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    random.seed(RANDOM_SEED)
    required_per_class = sum(SPLIT_SIZES.values())

    for split_name in SPLIT_SIZES:
        reset_dir(os.path.join(OUTPUT_ROOT, split_name))

    for class_name in CLASSES:
        source_class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(source_class_dir):
            raise FileNotFoundError(f"Khong tim thay thu muc nguon: {source_class_dir}")

        all_images = collect_images(source_class_dir)
        print(f"Tim thay {len(all_images)} anh trong '{source_class_dir}'")

        if len(all_images) < required_per_class:
            raise ValueError(
                f"Thu muc '{class_name}' chi co {len(all_images)} anh, can it nhat {required_per_class} anh"
            )

        random.shuffle(all_images)

        start_index = 0
        for split_name, split_size in SPLIT_SIZES.items():
            selected_images = all_images[start_index:start_index + split_size]
            start_index += split_size

            destination_class_dir = os.path.join(OUTPUT_ROOT, split_name, class_name)
            os.makedirs(destination_class_dir, exist_ok=True)

            print(
                f"Dang sao chep {len(selected_images)} anh '{class_name}' vao '{destination_class_dir}'"
            )
            for image_name in selected_images:
                source_path = os.path.join(source_class_dir, image_name)
                destination_path = os.path.join(destination_class_dir, image_name)
                shutil.copy2(source_path, destination_path)

    print("\nHoan thanh chia du lieu khong trung lap vao train/valid/test.")


if __name__ == "__main__":
    main()