# Visual Product Search Engine (DeepFashion Retrieval)

## Overview
This project builds an end-to-end visual product search engine for fashion image retrieval using the **DeepFashion In-Shop Clothes Retrieval dataset**. Given a query image, the system retrieves visually similar clothing items from a large gallery using deep learning–based embeddings and approximate nearest neighbor search.

## Key Features
- Clothing localization using **YOLOv8n fine-tuned on DeepFashion bounding boxes**
- Visual embeddings using **OpenCLIP ViT-B/32**
- Semantic enrichment using **BLIP-2 generated captions**
- Image-text fusion for improved retrieval performance
- Supervised contrastive fine-tuning of CLIP for better identity separation
- Scalable search using **HNSW approximate nearest neighbor indexing**
- Interactive **Streamlit UI** for end-to-end demo

## Pipeline
1. Dataset parsing (DeepFashion partitions + metadata)
2. Clothing detection and cropping (YOLOv8n)
3. Feature extraction (CLIP image encoder)
4. Caption generation (BLIP-2 for gallery images)
5. Embedding fusion (image + text)
6. Retrieval using HNSW + cosine similarity
7. Evaluation using Recall@K, NDCG@K, mAP@K

## Configurations
- **A**: Vision-only CLIP baseline
- **B**: CLIP + BLIP caption fusion (alpha weighting)
- **C**: Fine-tuned CLIP + fusion (best performance)

## Results
Best model (Configuration C, α = 0.7):
- Recall@15: **0.8549**
- NDCG@15: **0.4949**
- mAP@15: **0.3866**

## Tech Stack
- PyTorch
- OpenCLIP
- YOLOv8 (Ultralytics)
- BLIP-2 (Salesforce)
- FAISS / HNSW indexing
- Streamlit

## Dataset
DeepFashion In-Shop Clothes Retrieval Benchmark:
- Train / Query / Gallery splits
- Bounding box annotations
- Item-level metadata and descriptions

## Run Application
```bash
streamlit run app.py
