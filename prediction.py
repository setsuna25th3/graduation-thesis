import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from time import perf_counter
from datetime import datetime
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json

from model.pred_func import load_genconvit
from model.config import load_config

config = load_config()

def evaluate_images(root_dir, net, ed_weight, vae_weight, fp16):
    """Scans directory, predicts images and returns a JSON-like dict.
    Returned format: {"image": {"name":[], "pred":[], "klass":[], "pred_label":[], "correct_label":[]}}
    """
    result_json = {"image": {"name": [], "pred": [], "klass": [], "pred_label": [], "correct_label": []}}
    f_count = 0
    r_count = 0

    # 1. Load Model
    print(f"\n[INFO] Loading Model: {net}...")
    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)
    model.eval()

    device = next(model.parameters()).device
    print(f"[INFO] Model loaded on device: {device}")

    img_size = 224
    if config["model"].get("type") == "large":
        img_size = 384

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    count = 0

    print("\n[INFO] Starting prediction...")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(valid_extensions):
                continue

            img_path = os.path.join(dirpath, filename)
            count += 1

            lower_dir = dirpath.lower()
            if "real" in lower_dir or "original" in lower_dir or "0" in lower_dir:
                true_label = "REAL"
            elif "fake" in lower_dir or "manipulated" in lower_dir or "1" in lower_dir:
                true_label = "FAKE"
            else:
                true_label = "unknown"

            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                if fp16:
                    tensor = tensor.half()

                with torch.no_grad():
                    outputs = model(tensor)
                    outputs = outputs.view(-1, outputs.size(-1)).mean(dim=0, keepdim=True)
                    probs = F.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    prob_fake = probs[0][0].item()
                    predicted_label = "FAKE" if pred_idx == 0 else "REAL"

                if predicted_label == "FAKE":
                    f_count += 1
                else:
                    r_count += 1

                print(f"[{count}] Predict: {prob_fake:.4f} {predicted_label} \t\t(Fake: {f_count} | Real: {r_count}) | File: {filename}")

                result_json["image"]["name"].append(filename)
                result_json["image"]["pred"].append(float(prob_fake))
                result_json["image"]["klass"].append("uncategorized")
                result_json["image"]["pred_label"].append(predicted_label)
                result_json["image"]["correct_label"].append(true_label)

            except Exception as e:
                print(f"An error occurred while processing {filename}: {str(e)}")

    return result_json


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT Image Evaluation")
    parser.add_argument("--p", type=str, required=True, help="Path to the test image directory")
    parser.add_argument("--s", help="Model size type: tiny, large.", default="tiny")
    parser.add_argument("--e", nargs='?', const='genconvit_ed_inference', default=None, help="Weight for ed.")
    parser.add_argument("--v", '--value', nargs='?', const='genconvit_vae_inference', default=None, help="Weight for vae.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision (FP16)")

    args = parser.parse_args()
    path = args.p
    fp16 = args.fp16

    net = 'genconvit'
    ed_weight = None
    vae_weight = None
    
    if args.e and args.v:
        ed_weight = args.e
        vae_weight = args.v
    elif args.e:
        net = 'ed'
        ed_weight = args.e
    elif args.v:
        net = 'vae'
        vae_weight = args.v
        
    if args.s in ['tiny', 'large']:
        config["model"]["backbone"] = f"convnext_{args.s}"
        config["model"]["embedder"] = f"swin_{args.s}_patch4_window7_224"
        config["model"]["type"] = args.s
    
    return path, net, fp16, ed_weight, vae_weight


def main():
    start_time = perf_counter()
    
    # Parse arguments
    root_dir, net, fp16, ed_weight, vae_weight = gen_parser()
    
    if not os.path.exists(root_dir):
        print(f"[ERROR] Directory '{root_dir}' does not exist!")
        return
    # Run Evaluation
    result_json = evaluate_images(root_dir, net, ed_weight, vae_weight, fp16)

    # Save directory and calculate metrics (from JSON result)
    if result_json and result_json.get("image") and len(result_json["image"].get("name", []))>0:
        os.makedirs("result", exist_ok=True)
        curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M")

        print("\n" + "="*50)
        print("Evaluation metrics:")
        print("="*50)

        try:
            corrects = result_json["image"].get("correct_label", [])
            preds = result_json["image"].get("pred_label", [])
            valid_idx = [i for i, lbl in enumerate(corrects) if lbl != 'unknown']

            if len(valid_idx) > 0:
                y_true = [0 if corrects[i] == 'REAL' else 1 for i in valid_idx]
                y_pred = [0 if preds[i] == 'REAL' else 1 for i in valid_idx]
                y_scores = [result_json["image"]["pred"][i] for i in valid_idx]

                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                try:
                    auc = roc_auc_score(y_true, y_scores)
                except Exception:
                    auc = None

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                print(f"Total Images: {len(valid_idx)}")
                print(f"Accuracy    : {acc:.4f} ({(acc*100):.2f}%)")
                print(f"F1-Score    : {f1:.4f}")
                print(f"ROC-AUC     : {auc if auc is not None else 'N/A'}")
                print(f"FPR         : {fpr:.4f} (Real mistaken as Fake)")
                print(f"FNR         : {fnr:.4f} (Fake mistaken as Real)")
                print(f"Details     : TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            else:
                print("\n[WARNING] No ground truth labels found. Make sure your folders are named 'Real' and 'Fake'.")
        except Exception as e:
            print(f"Error calculating metrics: {e}")

        print("="*50 + "\n")
    else:
        print("[WARNING] No images found or processed in the directory.")

    end_time = perf_counter()
    print("--- Total processing time: {:.2f} seconds ---".format(end_time - start_time))

    # Export JSON results
    try:
        if result_json and result_json.get("image") and len(result_json["image"].get("name", []))>0:
            json_path = os.path.join("result", f"prediction_images_{net}_{curr_time}.json")
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(result_json, jf, ensure_ascii=False, indent=4)
            print(f"[OK] JSON results exported at: {json_path}")
    except Exception as e:
        print(f"Error exporting JSON: {e}")

if __name__ == "__main__":
    main()