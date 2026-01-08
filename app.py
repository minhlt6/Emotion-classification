import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import gdown

class HOGDescriptor:
    def __init__(self, img_size=(64, 64), cell_size=(8, 8), block_size=(2, 2), bins=9):
        self.img_size = img_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def process_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return img

    def compute_gradients(self, img):
        kernel_x = np.array([-1, 0, 1])
        kernel_y = np.array([-1, 0, 1]).T
        gx = cv2.filter2D(img, -1, kernel_x)
        gy = cv2.filter2D(img, -1, kernel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180 / np.pi)
        angle = angle % 180
        return magnitude, angle

    def compute_histograms(self, magnitude, angle):
        h, w = magnitude.shape
        cell_h, cell_w = self.cell_size
        n_cell_y = h // cell_h
        n_cell_x = w // cell_w
        histograms = np.zeros((n_cell_y, n_cell_x, self.bins))
        bin_width = 180 / self.bins

        for i in range(n_cell_y):
            for j in range(n_cell_x):
                cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ang = angle[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                for y in range(cell_h):
                    for x in range(cell_w):
                        mag = cell_mag[y, x]
                        ang = cell_ang[y, x]
                        bin_idx = int(ang // bin_width) % self.bins
                        next_bin_idx = (bin_idx + 1) % self.bins
                        weight = (ang % bin_width) / bin_width
                        histograms[i, j, bin_idx] += mag * (1 - weight)
                        histograms[i, j, next_bin_idx] += mag * weight
        return histograms

    def compute_block_normalization(self, histograms):
        n_cell_y, n_cell_x, _ = histograms.shape
        block_h, block_w = self.block_size
        n_block_y = n_cell_y - block_h + 1
        n_block_x = n_cell_x - block_w + 1
        normalized_blocks = []
        for i in range(n_block_y):
            for j in range(n_block_x):
                block = histograms[i:i+block_h, j:j+block_w].flatten()
                norm = np.sqrt(np.sum(block**2) + 1e-5)
                normalized_blocks.append(block / norm)
        return np.concatenate(normalized_blocks)

    def extract_features(self, img_array):
        processed_img = self.process_image(img_array)
        mag, ang = self.compute_gradients(processed_img)
        hist = self.compute_histograms(mag, ang)
        features = self.compute_block_normalization(hist)
        return features

MODEL_CONFIGS = {
    "Random Forest": {
        "id": "1PrrF8vO0xIBbcj8hkYHYQOoGHrr-bkqw", 
        "file": "rf_model.pkl"
    },
    "ID3": {
        "id": "1_JTMBw1rBzvNs8SKW_s-eaF0kAhhZWhz", 
        "file": "id3_model.pkl"
    },
    "CART": {
        "id": "1LeDg_XCMYGsr_WkM6fby7lcf0_W7Gk7c", 
        "file": "cart_model.pkl"
    },
    "KNN": {
        "id": "1HzvDgRDlhkt7LvhvqPtwT5g-AVwhmDfA", 
        "file": "knn_model.pkl"
    }
}

SELECTOR_FILENAME = 'selector.pkl'

@st.cache_resource
def load_all_models():
    loaded_models = {}
    selector = None
    if os.path.exists(SELECTOR_FILENAME):
        selector = joblib.load(SELECTOR_FILENAME)
    
    for name, config in MODEL_CONFIGS.items():
        file_path = config["file"]
        drive_id = config["id"]
        
        if not os.path.exists(file_path):
            url = f'https://drive.google.com/uc?id={drive_id}'
            try:
                gdown.download(url, file_path, quiet=False)
            except:
                st.warning(f"Lỗi tải {name}")
                continue

        try:
            if os.path.exists(file_path):
                loaded_models[name] = joblib.load(file_path)
        except Exception as e:
            st.error(f"Lỗi load {name}: {e}")

    return loaded_models, selector

def process_image_input(image_pil, selector):
    img_array = np.array(image_pil)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    hog_desc = HOGDescriptor(img_size=(64, 64), cell_size=(8, 8), block_size=(2, 2), bins=9)
    features = hog_desc.extract_features(img_array)
    features = features.reshape(1, -1)
    
    if selector:
        features = selector.transform(features)
    return features

st.set_page_config(layout="wide")
st.title("So sánh 4 thuật toán: ID3 - CART - RF - KNN")
st.markdown("Demo nhận diện cảm xúc sử dụng đặc trưng HOG.")

models, selector = load_all_models()

if not models:
    st.error("Chưa load được model nào.")
else:
    st.success(f"Đã load: {', '.join(models.keys())}")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Chọn ảnh cần dự đoán...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=250, caption="Ảnh đầu vào")

with col2:
    if uploaded_file is not None and models:
        if st.button("Dự đoán", type="primary"):
            with st.spinner('Đang xử lý...'):
                try:
                    features = process_image_input(image, selector)
                    
                    emotion_labels = {
                        0: "Angry", 
                        1: "Fear", 
                        2: "Happy", 
                        3: "Sad", 
                        4: "Surprise"
                    }

                    st.subheader("Kết quả dự đoán:")
                    res_cols = st.columns(4)
                    
                    for i, (model_name, model) in enumerate(models.items()):
                        pred_idx = model.predict(features)[0]
                        pred_text = emotion_labels.get(pred_idx, f"Lớp {pred_idx}")
                        
                        with res_cols[i % 4]: 
                            st.info(f"**{model_name}**")
                            st.metric(label="Cảm xúc", value=pred_text)
                            
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")