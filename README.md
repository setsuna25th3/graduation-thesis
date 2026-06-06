# GenConViT Deepfake Detection

GenConViT is a deepfake detection project for both images and videos. This repository provides the full workflow for training, inference, and result aggregation on common datasets such as DFDC, FaceForensics++, Celeb-DF, and DeepfakeTIMIT.

## Highlights

- Supports two model variants: `ed` and `vae`.
- Runs inference on a single image, an image folder, a single video, or a dataset directory.
- Exports predictions to JSON for metric analysis and ROC plotting.
- Saves checkpoints, metrics, and weights in a dedicated folder for easier management.

## Quick Structure

- `train.py`: trains the model.
- `prediction.py`: runs image inference.
- `my_predict.py`: runs video or dataset-based inference.
- `result_all.py`: aggregates results from the `result/` directory and plots ROC curves.
- `model/`: model definitions and loading utilities.
- `dataset/`: data loading utilities.

## Requirements

Python 3.10+ is recommended. A GPU is helpful for faster training, but the project can also run on CPU.

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

3. Prepare your data and model weights using the expected project structure.

## Data Layout

### Image Training

The training, validation, and test splits should follow this structure:

```text
data/
    train/
        fake/
        real/
    valid/
        fake/
        real/
    test/
        fake/
        real/
```

### Image Inference

The input directory can contain images and nested subfolders. Folder names should hint at the label, for example `real`, `fake`, `original`, or `manipulated`.

### Video Inference

For video data, keep the directory structure aligned with the target dataset. The script walks the directory tree recursively.

## Training

Basic training command:

```bash
python train.py --d <data_path> --m <ed|vae> --e <num_epochs> --b <batch_size>
```

Options:

- `--d`: path to the training data.
- `--m`: model variant, either `ed` or `vae`.
- `--e`: number of epochs.
- `--p`: pretrained checkpoint to continue training from.
- `--b`: batch size, defaulting to the value in the config.
- `--t`: run test evaluation after training if provided.

Examples:

```bash
python train.py --d sample_train_data --m vae --e 5 --t y
python train.py --d sample_train_data --m ed --e 5 --t y
```

After training, the model and metrics are saved in the `weight/` directory.

## Image Inference

Use `prediction.py` for images and image folders.

```bash
python prediction.py --p <image_folder> [--e <ed_weight_name>] [--v <vae_weight_name>] [--fp16] [--s tiny|large]
```

Options:

- `--p`: input image directory.
- `--e`: weight for the `ed` branch. If no value is provided, the script uses its default weight.
- `--v`: weight for the `vae` branch. If no value is provided, the script uses its default weight.
- `--fp16`: enable half-precision inference.
- `--s`: backbone size, either `tiny` or `large`.

Examples:

```bash
python prediction.py --p sample_prediction_data --e
python prediction.py --p sample_prediction_data --v
python prediction.py --p sample_prediction_data --e genconvit_ed_May_16_2024_10_18_09 --v genconvit_vae_May_16_2024_09_34_21 --fp16
```

Predictions are printed to the console, and when ground-truth labels are available through folder names, the script also computes metrics such as Accuracy, F1-score, and ROC-AUC.

## Video Inference

If you want to run video-based inference on datasets such as DFDC, FaceForensics++, or DeepfakeTIMIT, use `my_predict.py`.

```bash
python my_predict.py
```

This script is designed to read data from the dataset directories already supported by the project and export results as JSON.

## Aggregating Results

Once JSON files are available in the `result/` directory, run:

```bash
python result_all.py
```

The script will:

- read all JSON files in `result/`.
- compute ROC, AUC, F1-score, and accuracy.
- save the ROC plot to `img/roc_curve_result.png`.

## Notes

- Keep weight file names consistent when passing them to inference commands.
- Some dataset paths in the code are examples, so update them to match your machine before running.
- The project can run on CPU, but training and inference will be much slower.

## Sample Outputs

Example result files are stored in the `result/` directory.