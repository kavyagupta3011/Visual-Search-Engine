"""
eval_clip_seeds.py — Compare 3 fine-tuned CLIP models (different seeds)
for visual product search quality.

Ground truth: images sharing the same `item_id` in gallery_metadata.csv
are considered positive matches. A good model should retrieve them in top-K.

Metrics computed per model:
  • Recall@1, @5, @10   — is any same-item image in the top-K results?
  • MRR                 — mean reciprocal rank of the first correct hit
  • mAP@10             — mean average precision at 10
  • Intra-class cosine  — avg similarity between same-item embeddings (cohesion)
  • Inter-class cosine  — avg similarity between different-item embeddings (separation)

Usage:
  python eval_clip_seeds.py

Assumptions:
  • gallery_metadata.csv lives in the working directory
  • Model checkpoint filenames: clip_finetuned_16.pt, clip_finetuned_42.pt, clip_finetuned_99.pt
    (edit SEED_CHECKPOINTS below to match your actual filenames)
  • Gallery images are under ./gallery/ as per your app
  • Requires: torch, open_clip, pandas, numpy, Pillow, tqdm
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SEED_CHECKPOINTS = {
    "seed_16": "clip_finetuned_16.pt",
    "seed_42": "clip_finetuned_42.pt",
    "seed_99": "clip_finetuned_99.pt",
}

METADATA_PATH   = "gallery_metadata.csv"
GALLERY_ROOT    = "gallery"
CLIP_ARCH       = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE      = 64           # lower if you run out of VRAM
RECALL_AT       = [1, 5, 10]
MAP_AT          = 10
MIN_IMAGES_PER_ITEM = 2        # items with only 1 image can't be evaluated


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def resolve_path(row):
    """Mirror the path logic from app.py."""
    candidates = [
        os.path.join(GALLERY_ROOT, row["relative_path"]),
        os.path.join(GALLERY_ROOT, str(row["item_id"]),
                     os.path.basename(row["relative_path"])),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def load_model(checkpoint_path: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def embed_images(model, preprocess, image_paths: list, device: str) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, D)."""
    all_embs = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="  embedding", leave=False):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception:
                # Corrupted image → zero vector placeholder
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy().astype("float32"))

    return np.concatenate(all_embs, axis=0)


# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(embeddings: np.ndarray, item_ids: np.ndarray) -> dict:
    """
    For every image that has ≥1 other image of the same item_id,
    rank all other images by cosine similarity and compute retrieval metrics.
    """
    N = len(embeddings)
    # Full cosine similarity matrix (embeddings are already L2-normed)
    sim_matrix = embeddings @ embeddings.T          # (N, N)
    np.fill_diagonal(sim_matrix, -2.0)              # exclude self

    recall_hits  = {k: 0 for k in RECALL_AT}
    rr_scores    = []
    ap_scores    = []
    n_queries    = 0

    # Group indices by item_id for fast lookup
    item_to_indices = defaultdict(list)
    for idx, iid in enumerate(item_ids):
        item_to_indices[iid].append(idx)

    for q_idx in range(N):
        iid = item_ids[q_idx]
        positives = set(item_to_indices[iid]) - {q_idx}

        if not positives:
            continue    # singleton item — skip

        n_queries += 1
        ranked = np.argsort(-sim_matrix[q_idx])    # descending similarity

        # ── Recall@K ──
        for k in RECALL_AT:
            top_k = set(ranked[:k])
            if top_k & positives:
                recall_hits[k] += 1

        # ── MRR ──
        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in positives:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

        # ── mAP@MAP_AT ──
        hits, precision_sum = 0, 0.0
        for rank, idx in enumerate(ranked[:MAP_AT], start=1):
            if idx in positives:
                hits += 1
                precision_sum += hits / rank
        n_rel = min(len(positives), MAP_AT)
        ap_scores.append(precision_sum / n_rel if n_rel > 0 else 0.0)

    # ── Intra / Inter class similarity ──
    intra, inter = [], []
    sample_size = min(N, 2000)                      # cap for speed
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(N, size=sample_size, replace=False)

    for i in sample_idx:
        for j in sample_idx:
            if i == j:
                continue
            s = float(sim_matrix[i, j])             # diagonal already -2
            if s == -2.0:
                continue
            if item_ids[i] == item_ids[j]:
                intra.append(s)
            else:
                inter.append(s)

    return {
        **{f"Recall@{k}": recall_hits[k] / n_queries for k in RECALL_AT},
        "MRR":              float(np.mean(rr_scores)),
        f"mAP@{MAP_AT}":   float(np.mean(ap_scores)),
        "Intra-class sim":  float(np.mean(intra))  if intra  else float("nan"),
        "Inter-class sim":  float(np.mean(inter))  if inter  else float("nan"),
        "n_queries":        n_queries,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  CLIP Seed Eval   |   device={device}")
    print(f"{'='*60}\n")

    # ── Load metadata ──
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing {METADATA_PATH}")

    meta = pd.read_csv(METADATA_PATH)
    print(f"Metadata: {len(meta)} rows loaded.")

    # Resolve image paths
    meta["resolved_path"] = meta.apply(resolve_path, axis=1)
    missing = meta["resolved_path"].isna().sum()
    if missing:
        print(f"  ⚠  {missing} images not found on disk — skipping them.")

    meta = meta.dropna(subset=["resolved_path"]).reset_index(drop=True)

    # Filter to items with enough images to evaluate
    counts = meta["item_id"].value_counts()
    valid_items = counts[counts >= MIN_IMAGES_PER_ITEM].index
    eval_meta = meta[meta["item_id"].isin(valid_items)].reset_index(drop=True)

    print(f"  Evaluating on {len(eval_meta)} images "
          f"across {eval_meta['item_id'].nunique()} items "
          f"(≥{MIN_IMAGES_PER_ITEM} images each)\n")

    image_paths = eval_meta["resolved_path"].tolist()
    item_ids    = eval_meta["item_id"].to_numpy()

    results = {}

    for name, ckpt in SEED_CHECKPOINTS.items():
        print(f"── {name}  ({ckpt}) ──")

        if not os.path.exists(ckpt):
            print(f"  ❌  Checkpoint not found — skipping.\n")
            continue

        t0 = time.time()
        model, preprocess = load_model(ckpt, device)
        embeddings = embed_images(model, preprocess, image_paths, device)
        elapsed = time.time() - t0

        metrics = compute_metrics(embeddings, item_ids)
        metrics["embed_time_s"] = round(elapsed, 1)
        results[name] = metrics

        # Free VRAM before loading next model
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"  ✓  Done in {elapsed:.1f}s\n")

    # ── Results table ──
    if not results:
        print("No models evaluated.")
        return

    df = pd.DataFrame(results).T
    metric_cols = [f"Recall@{k}" for k in RECALL_AT] + \
                  ["MRR", f"mAP@{MAP_AT}", "Intra-class sim", "Inter-class sim",
                   "n_queries", "embed_time_s"]
    df = df[[c for c in metric_cols if c in df.columns]]

    # Format percentages
    pct_cols = [c for c in df.columns if "Recall" in c or "MRR" in c or "mAP" in c]
    for c in pct_cols:
        df[c] = df[c].apply(lambda x: f"{float(x)*100:.2f}%")

    sim_cols = [c for c in df.columns if "sim" in c]
    for c in sim_cols:
        df[c] = df[c].apply(lambda x: f"{float(x):.4f}")

    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(df.to_string())
    print()

    # ── Winner ──
    # Score = weighted sum: Recall@5 (40%) + MRR (35%) + mAP (25%)
    raw = pd.DataFrame(results).T
    scores = {}
    for name in raw.index:
        r = raw.loc[name]
        score = (
            0.40 * float(str(r.get(f"Recall@5",  0)).replace("%","")) +
            0.35 * float(str(r.get("MRR",         0)).replace("%","")) +
            0.25 * float(str(r.get(f"mAP@{MAP_AT}", 0)).replace("%",""))
        )
        scores[name] = score

    winner = max(scores, key=scores.get)
    print(f"  🏆  Recommended model: {winner}  (checkpoint: {SEED_CHECKPOINTS[winner]})")
    print(f"      Composite score: {scores[winner]:.4f}  "
          f"(0.40×Recall@5 + 0.35×MRR + 0.25×mAP@{MAP_AT})\n")

    # Save to CSV
    out_path = "eval_results.csv"
    pd.DataFrame(results).T.to_csv(out_path)
    print(f"  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
