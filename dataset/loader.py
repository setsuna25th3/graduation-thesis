import os
import torch
from torchvision import transforms, datasets
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    HueSaturationValue,
    GaussNoise,
    Sharpen,
    Emboss,
    RandomBrightnessContrast,
    OneOf,
    Compose,
)
import numpy as np
from PIL import Image


def strong_aug(p=0.5):
    # Apply a strong sequence of augmentations with a given probability p
    return Compose(
        [
            RandomRotate90(p=0.2),
            Transpose(p=0.2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf(
                [
                    GaussNoise(),
                ],
                p=0.2,
            ),
            ShiftScaleRotate(p=0.2),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.2,
            ),
            HueSaturationValue(p=0.2),
        ],
        p=p,
    )


def augment(aug, image):
    # Apply the defined albumentations pipeline to a given image
    return aug(image=image)["image"]


class Aug(object):
    # Wrapper class to integrate albumentations with torchvision transforms
    def __call__(self, img):
        aug = strong_aug(p=0.9)
        return Image.fromarray(augment(aug, np.array(img)))


def normalize_data(img_size=224):
    # Define mean and standard deviation for ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define transformations for each dataset split
    # Added transforms.Resize to ensure all images have the same dimensions before batching
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),  # Resize images to uniform size
                Aug(),                                    # Apply augmentations
                transforms.ToTensor(),                    # Convert PIL image to PyTorch tensor
                transforms.Normalize(mean, std)           # Normalize tensor values
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ]
        ),
        "vid": transforms.Compose([transforms.Normalize(mean, std)]),
    }


def load_data(data_dir="sample/", batch_size=4, img_size=224):
    # Load datasets from folders using ImageFolder and apply transformations
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), normalize_data(img_size)[x])
        for x in ["train", "valid", "test"]
    }

    # Store the number of images in each split
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}

    # Create dataloaders for batching, shuffling, and parallel data loading
    train_dataloaders = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_dataloaders = torch.utils.data.DataLoader(
        image_datasets["valid"],
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_dataloaders = torch.utils.data.DataLoader(
        image_datasets["test"],
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    dataloaders = {
        "train": train_dataloaders,
        "validation": validation_dataloaders,
        "test": test_dataloaders,
    }

    return dataloaders, dataset_sizes


def load_checkpoint(model, optimizer, filename=None):
    start_epoch = 0
    log_loss = 0
    
    # Check if the checkpoint file exists
    if filename and os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        
        # Restore state from the checkpoint
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_loss = checkpoint.get("min_loss", 0)  # Safe get in case min_loss is not in checkpoint
        
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, log_loss