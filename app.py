import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- Cấu hình trang ---
st.set_page_config(layout="wide", page_title="Nhận diện cảm xúc HOG - Final")

# ==========================================
# 1. CẤU HÌNH & LOAD MODEL
# ==========================================

# Tên file selector (BẮT BUỘC PHẢI CÓ để giảm chiều vector)
SELECTOR_FILENAME = 'selector.pkl'

# ID Google Drive của file selector.pkl (BẠN HÃY ĐIỀN ID CỦA BẠN VÀO ĐÂY NẾU CÓ)
# Nếu không, bạn phải upload file selector.pkl lên cùng thư mục với app.py
SELECTOR_DRIVE_ID = None  # Ví dụ: "1...ID_Cua_Ban..."

MODEL_CONFIGS = {
    "Random Forest": {"id": "1PrrF8vO0xIBbcj8hkYHYQOoGHrr-bkqw", "file": "rf_model.pkl"},
    "ID3": {"id": "1_JTMBw1rBzvNs8SKW_s-eaF0kAhhZWhz", "file": "id3_model.pkl"},
    "CART": {"id": "1LeDg_XCMYGsr_WkM6fby7lcf0_W7Gk7c", "file": "cart_model.pkl"},
    "KNN": {"id": "1HzvDgRDlhkt7LvhvqPtwT5g-AVwhmDfA", "file": "knn_model.pkl"}
}

@st.cache_resource
def load_resources():
    loaded_models = {}
    selector = None
    
    # 1. Tải và load Selector (QUAN TRỌNG)
    if not os.path.exists(SELECTOR_FILENAME) and SELECTOR_DRIVE_ID:
        url = f'https://drive.google.com/uc?id={SELECTOR_DRIVE_ID}'
        try:
            gdown.download(url, SELECTOR_FILENAME, quiet=True)
        except: pass
        
    if os.path.exists(SELECTOR_FILENAME):
        try:
            selector = joblib.load(SELECTOR_FILENAME)
        except Exception as e:
            st.error(f"Lỗi load selector.pkl: {e}")
    else:
        st.warning("⚠️ Không tìm thấy file 'selector.pkl'. Mô hình có thể bị lỗi kích thước (Shape Mismatch)!")

    # 2. Tải và load Models
    for name, config in MODEL_CONFIGS.items():
        file_path = config["file"]
        drive_id = config["id"]
        
        if not os.path.exists(file_path):
            url = f'https://drive.google.com/uc?id={drive_id}'
            try:
                gdown.download(url, file_path, quiet=True)
            except: pass

        if os.path.exists(file_path):
            try:
                loaded_models[name] = joblib.load(file_path)
            except Exception as e:
                st.error(f"Lỗi load {name}: {e}")
                
    return loaded_models, selector

# ==========================================
# 2. XỬ LÝ ẢNH & HOG (Đã cập nhật chuẩn 64x64)
# ==========================================
class HOGDescriptor:
    def __init__(self, img_size=(64, 64), cell_size=(8, 8), block_size=(2, 2), bins=9):
        self.img_size = img_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def process_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Resize về 64x64 (BẮT BUỘC như lúc train)
        img = cv2.resize(img, self.img_size)

        # 2. Cân bằng sáng (Histogram Equalization) -> Giúp ảnh Webcam rõ nét như ảnh train
        img = cv2.equalizeHist(img)
        
        # 3. Làm mờ nhẹ để khử nhiễu webcam
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 4. Chuẩn hóa về 0-1
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

# ==========================================
# 3. HÀM CẮT MẶT (TIGHT CROP)
# ==========================================
def detect_face(image_array):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None
    
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    # --- CẮT SÁT (ZOOM IN) ---
    # Thu hẹp 15% viền để loại bỏ tóc/cổ, chỉ lấy nét mặt chính
    zoom_ratio = 0.15 
    offset_x = int(w * zoom_ratio)
    offset_y = int(h * zoom_ratio)
    
    new_x = x + offset_x
    new_y = y + offset_y
    new_w = w - (2 * offset_x)
    new_h = h - (2 * offset_y)
    
    if new_w > 0 and new_h > 0:
        best_face = image_array[new_y:new_y+new_h, new_x:new_x+new_w]
        return best_face, (new_x, new_y, new_w, new_h)
    else:
        return image_array[y:y+h, x:x+w], (x, y, w, h)

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.title("Phân loại cảm xúc: ID3 - CART - RF - KNN")
st.markdown("Quy trình: Detect Face -> Crop -> HOG -> **Feature Selection (Giảm chiều)** -> Predict")

models, selector = load_resources()

if selector:
    st.success(f"✅ Đã tải Selector: {type(selector).__name__} (Sẵn sàng giảm chiều vector)")
else:
    st.error("❌ Chưa tải được file selector.pkl. Vui lòng kiểm tra!")

col1, col2 = st.columns([1, 1.5])
input_image_pil = None

with col1:
    st.subheader("1. Nhập ảnh")
    tab_upload, tab_cam = st.tabs(["📁 Upload", "📷 Camera"])
    
    with tab_upload:
        uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            input_image_pil = Image.open(uploaded_file)
    with tab_cam:
        cam_image = st.camera_input("Chụp ảnh")
        if cam_image:
            input_image_pil = Image.open(cam_image)

    if input_image_pil:
        input_array = np.array(input_image_pil)
        if len(input_array.shape) == 3 and input_array.shape[2] == 4:
            input_array = input_array[..., :3]

        st.info("Đang tìm khuôn mặt...")
        face_img, coords = detect_face(input_array)

        if face_img is not None:
            st.image(face_img, caption="Khuôn mặt đã cắt (Input cho Model)", width=150)
            st.session_state['face_to_process'] = face_img
        else:
            st.warning("⚠️ Không tìm thấy khuôn mặt rõ ràng. Dùng toàn bộ ảnh.")
            st.session_state['face_to_process'] = input_array

with col2:
    st.subheader("2. Kết quả dự đoán")
    
    if 'face_to_process' in st.session_state and input_image_pil is not None:
        if st.button("Chạy dự đoán", type="primary"):
            face_to_analyze = st.session_state['face_to_process']
            
            with st.spinner('Đang xử lý...'):
                try:
                    # 1. Trích xuất đặc trưng HOG
                    hog_desc = HOGDescriptor() # Mặc định 64x64
                    features = hog_desc.extract_features(face_to_analyze)
                    features = features.reshape(1, -1)
                    
                    st.write(f"Số lượng đặc trưng gốc: **{features.shape[1]}**")

                    # 2. GIẢM CHIỀU (FEATURE SELECTION) - QUAN TRỌNG
                    if selector:
                        features = selector.transform(features)
                        st.write(f"Số lượng đặc trưng sau khi giảm: **{features.shape[1]}**")
                    else:
                        st.error("Thiếu selector.pkl, không thể giảm chiều đặc trưng -> Có thể gây lỗi!")

                    # 3. Dự đoán
                    emotion_labels = {0: "Giận dữ 😡", 1: "Sợ hãi 😱", 2: "Vui vẻ 😄", 3: "Buồn 😢", 4: "Ngạc nhiên 😲"}
                    
                    st.write("---")
                    res_cols = st.columns(2)
                    for i, (name, model) in enumerate(models.items()):
                        try:
                            pred = model.predict(features)[0]
                            label = emotion_labels.get(pred, str(pred))
                            with res_cols[i % 2]:
                                st.success(f"**{name}**: {label}")
                        except Exception as e:
                             with res_cols[i % 2]:
                                st.error(f"{name} lỗi: {e}")

                except Exception as e:
                    st.error(f"Lỗi chung: {e}")