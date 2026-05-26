import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
from tqdm import tqdm
from dataset.loader import normalize_data
from .config import load_config
from .genconvit import GenConViT
# removed video-specific imports (decord, glob) per request
from PIL import Image
from time import perf_counter

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_genconvit(config, net, ed_weight, vae_weight, fp16):    
    model = GenConViT(
        config,
        ed= ed_weight,
        vae= vae_weight, 
        net=net,
        fp16=fp16
    )

    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"
    
    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )
        

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(
                    face_image, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


# video frame extraction functions removed — this module now focuses on image prediction


def is_image(img_path):
    return os.path.isfile(img_path) and img_path.lower().endswith(tuple([".jpg", ".jpeg", ".png", ".webp"]))


def df_face_from_image(img_path):
    try:
        im = Image.open(img_path).convert("RGB")
        arr = np.asarray(im)
    except Exception:
        return []

    face, count = face_rec([arr])
    return preprocess_frame(face) if count > 0 else []


def predict_img(img, model, fp16, result, klass, count=0, accuracy=-1, correct_label="unknown"):
    count += 1
    print(f"\n\n{str(count)} Loading image... {img}")
    start_time = perf_counter()

    df = df_face_from_image(img)
    if fp16 and hasattr(df, "half"):
        df = df.half()

    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )

    result = store_result(
        result, os.path.basename(img), y, y_val, klass, correct_label, None, result_type="image"
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}")

    end_time = perf_counter()
    print("\n\n image predict--- %s seconds ---" % (end_time - start_time))

    return result, accuracy, count, [y, y_val]


# video detection helpers removed


def set_result():
    # Return a result container for image predictions only.
    return {
        "image": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result,
    filename,
    y,
    y_val,
    klass,
    correct_label=None,
    compression=None,
    result_type="image",
):
    # result_type currently supports only "image". If callers pass another
    # type, create a generic container for it.
    if result_type not in result:
        result[result_type] = {"name": [], "pred": [], "klass": [], "pred_label": [], "correct_label": []}

    result[result_type]["name"].append(filename)
    result[result_type]["pred"].append(y_val)
    result[result_type]["klass"].append(klass.lower())
    result[result_type]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result[result_type]["correct_label"].append(correct_label)

    # Compression is a video-specific field; ignore it for image-only results.
    return result