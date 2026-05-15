"""
app.py — Streamlit Demo for Visual Product Search Engine
Run: streamlit run app.py
"""

import streamlit as st #type:ignore
import torch  #type:ignore
import hnswlib
import pandas as pd
import open_clip  #type:ignore
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO  #type:ignore

st.set_page_config(layout="wide", page_title="👗 Visual Product Search", page_icon="👗")



@st.cache_resource
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load YOLO




    ########## TO CHNAHEGEEEEEEEEEEEEEEEEEEEEEEEEE
    yolo = YOLO("best.pt")
    ##################################










    
    # 2. Load Fine-Tuned CLIP (Seed 16)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    
    if not os.path.exists("clip_finetuned_16.pt"):
        st.error("❌ Missing 'clip_finetuned_16.pt'! Put it in the same folder as app.py.")
        st.stop()
        
    clip_model.load_state_dict(torch.load("clip_finetuned_16.pt", map_location=device))
    clip_model = clip_model.to(device).eval()
    
    # 3. Load HNSW Index (Config C, Alpha 0.7, Seed 16)
    index = hnswlib.Index(space="cosine", dim=512)
    if not os.path.exists("index_C_07_16.bin"):
        st.error("❌ Missing 'index_C_07_16.bin'! Put it in the same folder as app.py.")
        st.stop()
        
    index.load_index("index_C_07_16.bin")
    
    # 4. Load Metadata
    metadata = pd.read_csv("gallery_metadata.csv")
    
    return yolo, clip_model, clip_preprocess, index, metadata, device



clothing_choice = st.selectbox(
    "Clothing Type",
    ["All", "Top", "Bottom", "Full Body"]
)

TYPE_MAP = {
    "All": None,
    "Top": 1,
    "Bottom": 2,
    "Full Body": 3
}

requested_type = TYPE_MAP[clothing_choice]



############################################################################
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
#########################################################################



def get_image_embedding(clip_model, clip_preprocess, pil_image, device):
    tensor = clip_preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


# ─── UI ───────────────────────────────────────────────────────────
st.title("👗 Visual Product Search Engine")
st.markdown("Upload a clothing image — the system will find visually and semantically similar products from the catalog.")

with st.spinner("Loading models into memory..."):
    yolo, clip_model, clip_preprocess, index, metadata, device = load_assets()

uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("---")
    st.subheader("Step 1: YOLO Product Detection")
    cropped_img, was_cropped = crop_with_yolo(
    yolo,
    original_img,
    requested_type
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="📷 Original Image", use_column_width=True)
    with col2:
        if was_cropped:
            st.image(cropped_img, caption="✂️ YOLO Cropped Region", use_column_width=True)
            st.success("YOLO successfully isolated the primary clothing item.")
        else:
            st.image(original_img, caption="No confident crop found", use_column_width=True)
            st.warning("YOLO didn't find a confident bounding box. Proceeding with the original image.")
    
    st.markdown("---")
    st.subheader("Step 2: Search Parameters")
    
    use_image = original_img
    if was_cropped:
        confirm = st.radio(
            "Confirm which image to send to the search engine:", 
            ["✅ Use YOLO Crop (Recommended)", "🔄 Use Original Image"]
        )
        if "Crop" in confirm:
            use_image = cropped_img
            
    K = st.slider("Number of results to retrieve (K)", min_value=3, max_value=15, value=5)
    


    if st.button("🔍 Search Similar Products", type="primary", use_container_width=True):
        with st.spinner("Embedding query and searching HNSW index..."):
            query_emb = get_image_embedding(clip_model, clip_preprocess, use_image, device)
            search_k = max(K * 5, 20)

            labels, distances = index.knn_query(query_emb, k=search_k)

            filtered_labels = []
            filtered_distances = []

            for lbl, dist in zip(labels[0], distances[0]):

                row = metadata.iloc[int(lbl)]

                if (
                    requested_type is None
                    or row.get("clothes_type") == requested_type
                ):
                    filtered_labels.append(lbl)
                    filtered_distances.append(dist)

                if len(filtered_labels) >= K:
                    break

            labels = [filtered_labels]
            distances = [filtered_distances]



            
        st.markdown("---")
        st.subheader(f"Step 3: Top {K} Matches Found")
        
        # PRO UI FIX: Grid Layout so K=15 doesn't crush the screen!
        cols_per_row = 5
        num_results = len(labels[0])

        for row_idx in range(0, num_results, cols_per_row):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                rank = row_idx + col_idx
                if rank >= num_results:
                    break # We've displayed all K results
                    
                label = labels[0][rank]
                dist = distances[0][rank]
                row = metadata.iloc[int(label)]
                
                # Robust path hunting
                path_candidates = [
                    f"gallery/{row['relative_path']}", 
                    f"gallery/{row['item_id']}/{os.path.basename(row['relative_path'])}"
                ]
                found_path = next((p for p in path_candidates if os.path.exists(p)), None)
                
                with cols[col_idx]:
                    st.markdown(f"**Rank #{rank + 1}**")
                    if found_path: 
                        st.image(Image.open(found_path), use_column_width=True)
                    else: 
                        st.error("Image missing locally")
                        
                    similarity = 1 - dist
                    color = "green" if similarity > 0.8 else "orange" if similarity > 0.6 else "red"
                    
                    st.markdown(f"<h4 style='color:{color}; margin-bottom:0px;'>Score: {similarity:.4f}</h4>", unsafe_allow_html=True)
                    st.caption(f"**ID:** `{row['item_id']}`\n\n📝 {row.get('caption', '')[:65]}...")