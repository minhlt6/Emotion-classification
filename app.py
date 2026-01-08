import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image, ImageOps
import os
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

MODEL_PATH = 'random_forest_model.pkl' 
SELECTOR_PATH = 'selector.pkl'         

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)

        if os.path.exists(SELECTOR_PATH):
            selector = joblib.load(SELECTOR_PATH)
        else:
            selector = None
            st.warning("Không tìm thấy file selector.pkl")
            
        return model, selector
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, None


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

# --- 5. GIAO DIỆN STREAMLIT ---
st.title("Phân loại Cảm xúc ")
st.write("Sử dụng HOG và các thuật toán học máy để phân loại cảm xúc từ ảnh.")

model, selector = load_resources()

if model:
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, width=200, caption="Ảnh đã tải lên")
        
        if st.button("Dự đoán"):
            with st.spinner('Đang xử lý...'):
                # Xử lý ảnh
                try:
                    features = process_image_input(image, selector)
                    
                    # Dự đoán
                    prediction = model.predict(features)[0]
                    
                    # Mapping nhãn (Bạn sửa lại theo đúng nhãn của mình)
                    emotion_labels = {
                        0: "Angry", 
                        1: "Fear", 
                        2: "Happy", 
                        3: "Sad", 
                        4: "Surprise"
                    }
                    result_text = emotion_labels.get(prediction, f"Lớp {prediction}")
                    
                    st.success(f"Kết quả: **{result_text}**")
                    
                except ValueError as ve:
                    st.error(f"Lỗi kích thước dữ liệu: {ve}")
                    st.info("Gợi ý: Có thể bạn chưa load file `selector.pkl` (VarianceThreshold) nên vector đầu vào (1764) không khớp với model (392).")
                except Exception as e:
                    st.error(f"Lỗi không xác định: {e}")