"""
eval.py — Batch Evaluation Script
Runs Config A, Config B (2 alphas), Config C (2 alphas) across seeds
Roll numbers: 16, 34, 59
"""

import os, random, torch, hnswlib #type: ignore
import pandas as pd, numpy as np
import open_clip  #type: ignore
from ultralytics import YOLO  #type: ignore
from PIL import Image
from pathlib import Path
from tqdm import tqdm

BASE_PATH =  "/kaggle/input/datasets/divyasiddavatam/visualrecognition"
QUERY_DIR = "/kaggle/input/datasets/divyasiddavatam/visualrecognition/Divya/PleaseBeCorrect/query"
GALLERY_METADATA = f"{BASE_PATH}/gallery_metadata.csv"
K_VALUES = [5, 10, 15]
SEEDS = [16, 34, 59]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_query_data(query_dir, metadata):
    query_data = []

    for item_folder in Path(query_dir).iterdir():
        if not item_folder.is_dir():
            continue

        gt_id = item_folder.name

        meta_rows = metadata[metadata["item_id"] == gt_id]

        if len(meta_rows) == 0:
            continue

        requested_type = meta_rows.iloc[0]["clothes_type"]

        for img_file in sorted(item_folder.glob("*.jpg")):
            query_data.append(
                (str(img_file), gt_id, requested_type)
            )

    return query_data



def crop_with_yolo(yolo_model, pil_image, requested_type):

    YOLO_CLASS_MAP = {
        1: 0,   # upper-body
        2: 1,   # lower-body
        3: 2    # full-body
    }

    requested_yolo_class = (
    YOLO_CLASS_MAP[requested_type]
    if requested_type is not None
    else None
)

    results = yolo_model(pil_image, verbose=False)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return pil_image, False

    matching_boxes = [
        b for b in boxes
        if float(b.conf) > 0.5
        and (
            requested_yolo_class is None
            or int(b.cls[0]) == requested_yolo_class
        )
    ]

    if not matching_boxes:
        return pil_image, False

    best = max(matching_boxes, key=lambda b: float(b.conf))

    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return pil_image, False

    return (
        pil_image.crop((
            max(0, x1),
            max(0, y1),
            min(pil_image.width, x2),
            min(pil_image.height, y2)
        )),
        True
    )



def get_image_embedding(clip_model, clip_preprocess, pil_image, device):
    tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def compute_all_metrics(retrieved_ids, gt_id, total_relevant, k_values):
    out = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        recall = 1 if gt_id in top_k else 0
        dcg = sum(
        1.0 / np.log2(i + 2)
        for i, r in enumerate(top_k)
        if r == gt_id
        )

        ideal_hits = min(total_relevant, k)

        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0


        hits, ap = 0, 0.0
        for i, r in enumerate(top_k):
            if r == gt_id: hits += 1; ap += hits / (i + 1)
        normalizer = min(total_relevant, k)
        out[k] = {
            "recall": recall,
            "ndcg": ndcg,
            "map": ap / normalizer if normalizer > 0 else 0.0
        }
            


    return out

def evaluate_config(config_name, query_data, index, metadata, clip_model, clip_preprocess, yolo_model, device, k_values, seed):
    set_seed(seed)
    gallery_id_counts = metadata["item_id"].value_counts().to_dict()
    results = {k: {"recall": [], "ndcg": [], "map": []} for k in k_values}
    successful = 0
    skipped = 0
    error_count = 0

    for img_path, gt_id, requested_type in tqdm(query_data, desc=f"{config_name} | seed={seed}"):
        try:
            img = Image.open(img_path).convert("RGB")
            emb = get_image_embedding(clip_model, clip_preprocess, crop_with_yolo(yolo_model, img, requested_type)[0], device)
            search_k = 200

            labels, distances = index.knn_query(emb, k=search_k)

            filtered_labels = []

            for lbl in labels[0]:

                row = metadata.iloc[int(lbl)]

                if (
                    requested_type is None
                    or row.get("clothes_type") == requested_type
                ):
                    filtered_labels.append(lbl)

                if len(filtered_labels) >= max(k_values):
                    break

            retrieved_ids = [
                metadata.iloc[int(l)]["item_id"]
                for l in filtered_labels
            ]

            if len(retrieved_ids) < max(k_values):
                skipped += 1
                continue

            m = compute_all_metrics(retrieved_ids, gt_id, gallery_id_counts.get(gt_id, 1), k_values)
            for k in k_values:
                results[k]["recall"].append(m[k]["recall"])
                results[k]["ndcg"].append(m[k]["ndcg"])
                results[k]["map"].append(m[k]["map"])
            successful += 1
        except Exception as e:
            error_count += 1
            print(f"ERROR on {img_path}: {e}")
            continue

    total = successful + skipped + error_count

    print(f"\n[{config_name}]")
    print(f"Successful queries : {successful}")
    print(f"Skipped queries    : {skipped}")
    print(f"Errored queries    : {error_count}")
    print(f"Total processed    : {total}\n")

    
    return results

def print_final_table(all_collected_results, k_values):
    """Prints everything in one massive, unified table exactly like the screenshot"""
    print("\n\n" + "="*85)
    print(f"{'Config':<20} | {'K':<4} | {'Recall@K':<18} | {'NDCG@K':<18} | {'mAP@K':<18}")
    print("="*85)
    
    for config_name, all_seed_results in all_collected_results.items():
        for k in k_values:
            r_mean = np.mean([np.mean(s[k]["recall"]) for s in all_seed_results])
            r_std  = np.std([np.mean(s[k]["recall"]) for s in all_seed_results])
            n_mean = np.mean([np.mean(s[k]["ndcg"]) for s in all_seed_results])
            n_std  = np.std([np.mean(s[k]["ndcg"]) for s in all_seed_results])
            m_mean = np.mean([np.mean(s[k]["map"]) for s in all_seed_results])
            m_std  = np.std([np.mean(s[k]["map"]) for s in all_seed_results])
            
            # Only print the config name on the first line (K=5) to keep it clean
            name_label = config_name if k == 5 else ""
            
            print(f"{name_label:<20} | {k:<4} | {r_mean:.4f}±{r_std:.4f}   | {n_mean:.4f}±{n_std:.4f}   | {m_mean:.4f}±{m_std:.4f}")
        print("-" * 85)
    print("="*85 + "\n")

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"




    yolo_model = YOLO(f"{BASE_PATH}/best.pt")




    clip_model_base, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model_base = clip_model_base.to(device).eval()
    
    metadata = pd.read_csv(GALLERY_METADATA)
    query_data = load_query_data(QUERY_DIR, metadata)
    index = hnswlib.Index(space="cosine", dim=512)

    master_results = {}

    # --- CONFIG A & B (Static Base Models) ---
    static_configs = [
        ("A - image only", "index_A.bin"), 
        ("B α=0.7", "index_B_07.bin"), 
        ("B α=0.5", "index_B_05.bin")
    ]
    
    for name, bin_file in static_configs:
        index.load_index(f"{BASE_PATH}/{bin_file}"); index.set_ef(50)
        # Duplicate base results 3 times so the standard deviation formats correctly as ±0.0000
        res = evaluate_config(name, query_data, index, metadata, clip_model_base, clip_preprocess, yolo_model, device, K_VALUES, 16)
        master_results[name] = [res, res, res]

    # --- CONFIG C (Dynamic Seed Loading) ---
    for alpha in ["07", "05"]:
        alpha_label = f"C α=0.{alpha[-1]}"
        master_results[alpha_label] = []
        for seed in SEEDS:
            clip_ft, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            clip_ft.load_state_dict(torch.load(f"{BASE_PATH}/clip_finetuned_{seed}.pt", map_location=device))
            clip_ft = clip_ft.to(device).eval()
            
            index.load_index(f"{BASE_PATH}/index_C_{alpha}_{seed}.bin"); index.set_ef(50)
            res = evaluate_config(f"{alpha_label} (Seed {seed})", query_data, index, metadata, clip_ft, clip_preprocess, yolo_model, device, K_VALUES, seed)
            master_results[alpha_label].append(res)

    # Print the master table!
    print_final_table(master_results, K_VALUES)

if __name__ == "__main__":
    run_evaluation()