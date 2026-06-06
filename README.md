# Deepfake Image Detection

Dataset download: [Google Drive link](https://drive.google.com/drive/folders/1s_HMAa7mC_0lyKJTIOVkigA8zQYzv1dH?usp=drive_link)

This repository is focused on image-based deepfake prediction. The main entry point is `prediction.py`, which scans an image folder, predicts each image as real or fake, and exports the results to JSON.

## What It Does

- Predicts deepfake labels for images in a folder tree.
- Supports two inference modes: `ed`, `vae`, or both together.
- Saves prediction results in the `result/` directory.
- Computes Accuracy, F1-score, ROC-AUC, FPR, and FNR when ground-truth labels can be inferred from folder names.

## Project Files

- `prediction.py`: image prediction script.
- `model/`: model definitions, config, and loading helpers.
- `result/`: exported JSON prediction files.
- `img/`: generated plots and figures.
- `requirements.txt`: Python dependencies.

## Requirements

Python 3.10+ is recommended.

Main packages listed in `requirements.txt`:

- `opencv-python`
- `face-recognition==1.3.0`
- `albumentations==1.3.0`
- `decord==0.6.0`
- `timm==0.6.5`

You will also need `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, and the PyTorch dependencies that match your CUDA or CPU environment.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/setsuna25th3/graduation-thesis.git
cd graduation-thesis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare an input folder that contains images.

## Input Folder Format

The script walks through the target directory recursively and processes these image formats:

```text
.png
.jpg
.jpeg
.webp
```

For automatic metric calculation, use folder names that indicate the label, such as:

```text
real
fake
original
manipulated
```

If a label cannot be inferred from the folder name, the script marks it as `unknown`.

## Usage

Run image prediction with:

```bash
python prediction.py --p <image_folder> [--e <ed_weight_name>] [--v <vae_weight_name>] [--fp16] [--s tiny|large]
```

Options:

- `--p`: input image directory.
- `--e`: weight for the `ed` branch. If a value is provided, the script uses that checkpoint.
- `--v`: weight for the `vae` branch. If a value is provided, the script uses that checkpoint.
- `--fp16`: enable half-precision inference.
- `--s`: backbone size, either `tiny` or `large`.

Examples:

```bash
python prediction.py --p sample_prediction_data --e
python prediction.py --p sample_prediction_data --v
python prediction.py --p sample_prediction_data --e ed_weight_name --v vae_weight_name --fp16
```

If both `--e` and `--v` are provided, the script runs the combined model mode. If only one of them is provided, it switches to the corresponding branch.

## Output

After prediction, the script:

- prints per-image results to the console.
- counts predicted real and fake samples.
- computes metrics when ground-truth labels are available.
- writes a JSON file named like `result/prediction_images_<mode>_<timestamp>.json`.

## Notes

- Keep weight file names consistent when passing them to inference commands.
- Some example checkpoint names in the code are defaults for local testing; replace them with your own weights when needed.
- The project can run on CPU, but inference will be slower than on GPU.