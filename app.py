import streamlit as st
import joblib
import numpy as np
import os
import gdown  # Thư viện tải file từ Drive
from PIL import Image, ImageOps
from skimage.feature import hog

# --- 1. CẤU HÌNH (BẮT BUỘC KHỚP VỚI LÚC TRAIN) ---
IMAGE_SIZE = (64, 64)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2'

# --- 2. CẤU HÌNH GOOGLE DRIVE ID ---
# Dán ID file model >100MB của bạn vào đây
MODEL_FILE_ID = '1JqghqUlS4e1PEsmndY-0RkTuCfjEUZTA'  # <--- THAY ID CỦA BẠN VÀO ĐÂY
MODEL_FILENAME = 'random_forest_model.pkl'

# --- 3. HÀM TẢI VÀ LOAD MODEL ---
@st.cache_resource
def load_resources():
    # A. Tải file Model nặng từ Drive về (nếu chưa có)
    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        try:
            st.info("Đang tải model > 100MB từ Google Drive... Vui lòng chờ 1-2 phút...")
            gdown.download(url, MODEL_FILENAME, quiet=False)
            st.success("Tải model thành công!")
        except Exception as e:
            st.error(f"Lỗi tải model: {e}")
            return None, None, None

    # B. Load các file (Model nặng + Scaler nhẹ + Class names nhẹ)
    try:
        # Load Model vừa tải
        model = joblib.load(MODEL_FILENAME)
        
        # Load Scaler và Class name (Những file này nhẹ, up thẳng lên GitHub nên load bình thường)
        scaler = joblib.load('scaler.pkl')
        class_names = joblib.load('class_names.pkl')
        
        return model, scaler, class_names
        
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None, None, None

# --- 4. HÀM XỬ LÝ ẢNH (GIỮ NGUYÊN) ---
def process_image(image_input, scaler):
    image = ImageOps.fit(image_input, IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    
    hog_features = hog(img_array,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       block_norm=HOG_BLOCK_NORM,
                       visualize=False,
                       feature_vector=True)
    
    features_reshaped = hog_features.reshape(1, -1)
    features_scaled = scaler.transform(features_reshaped)
    
    return features_scaled

# --- 5. GIAO DIỆN ---
st.title("Phân loại ảnh (Model > 100MB)")

model, scaler, class_names = load_resources()

if model and scaler:
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=250)
        
        if st.button('Dự đoán'):
            final_data = process_image(image, scaler)
            pred_idx = model.predict(final_data)[0]
            
            # Xử lý hiển thị tên lớp
            try:
                result_name = class_names[pred_idx]
            except:
                result_name = str(pred_idx)
                
            st.header(f"Kết quả: {result_name}")