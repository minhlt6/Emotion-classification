import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- Cấu hình trang ---
st.set_page_config(layout="wide", page_title="Nhận diện cảm xúc HOG")

# ==========================================
# 1. HÀM XỬ LÝ ẢNH & HOG
# ==========================================
class HOGDescriptor:
    def __init__(self, img_size=(64, 64), cell_size=(8, 8), block_size=(2, 2), bins=9):
        self.img_size = img_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def process_image(self, img):
        # Lưu ý: img ở đây là ảnh đã được cắt mặt
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

# ==========================================
# 2. HÀM CẮT MẶT 
def detect_face(image_array):
    """
    Hàm phát hiện và cắt khuôn mặt lớn nhất trong ảnh.
    Input: Numpy array (RGB)
    Output: Numpy array (RGB) chứa khuôn mặt đã cắt, hoặc None nếu không tìm thấy.
    """
    # Load mô hình nhận diện khuôn mặt có sẵn của OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Chuyển sang ảnh xám để detect
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None # Không tìm thấy mặt
    
    # Nếu tìm thấy nhiều mặt, lấy mặt có diện tích lớn nhất 
    max_area = 0
    best_face = None
    coords = None
    
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            # Cắt ảnh gốc theo tọa độ (lưu ý ảnh gốc là RGB)
            best_face = image_array[y:y+h, x:x+w]
            coords = (x, y, w, h)
            
    return best_face, coords

# ==========================================
# 3. QUẢN LÝ MODEL
# ==========================================
MODEL_CONFIGS = {
    "Random Forest": {"id": "1PrrF8vO0xIBbcj8hkYHYQOoGHrr-bkqw", "file": "rf_model.pkl"},
    "ID3": {"id": "1_JTMBw1rBzvNs8SKW_s-eaF0kAhhZWhz", "file": "id3_model.pkl"},
    "CART": {"id": "1LeDg_XCMYGsr_WkM6fby7lcf0_W7Gk7c", "file": "cart_model.pkl"},
    "KNN": {"id": "1HzvDgRDlhkt7LvhvqPtwT5g-AVwhmDfA", "file": "knn_model.pkl"}
}

SELECTOR_FILENAME = 'selector.pkl'

@st.cache_resource
def load_all_models():
    loaded_models = {}
    selector = None
    
    if os.path.exists(SELECTOR_FILENAME):
        try:
            selector = joblib.load(SELECTOR_FILENAME)
        except: pass
    
    for name, config in MODEL_CONFIGS.items():
        file_path = config["file"]
        drive_id = config["id"]
        
        if not os.path.exists(file_path):
            url = f'https://drive.google.com/uc?id={drive_id}'
            try:
                gdown.download(url, file_path, quiet=True)
            except:
                st.warning(f"Không thể tải model {name}")
                continue
             
        try:
            if os.path.exists(file_path):
                loaded_models[name] = joblib.load(file_path)
        except Exception as e:
            st.error(f"Lỗi load {name}: {e}")

    return loaded_models, selector

# Hàm trích xuất đặc trưng từ ảnh đã cắt
def extract_features_from_face(face_img_array, selector):
    hog_desc = HOGDescriptor(img_size=(64, 64), cell_size=(8, 8), block_size=(2, 2), bins=9)
    features = hog_desc.extract_features(face_img_array)
    features = features.reshape(1, -1)
    if selector:
        features = selector.transform(features)
    return features

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.title("Phân loại cảm xúc: ID3 - CART - RF - KNN")
st.markdown("Có tích hợp: **Tự động cắt khuôn mặt** trước khi dự đoán.")

models, selector = load_all_models()

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

    # Nếu có ảnh đầu vào
    if input_image_pil:
        # Hiển thị ảnh gốc
        st.image(input_image_pil, caption="Ảnh gốc", width=300)
        
        # Chuyển sang array
        input_array = np.array(input_image_pil)
        if len(input_array.shape) == 3 and input_array.shape[2] == 4:
            input_array = input_array[..., :3] # Bỏ kênh alpha

        # --- GIAI ĐOẠN CẮT MẶT ---
        st.info("Đang tìm khuôn mặt...")
        face_img, coords = detect_face(input_array)

        if face_img is not None:
            # Vẽ hình chữ nhật lên ảnh gốc để minh họa (Optional)
            x, y, w, h = coords
            cv2.rectangle(input_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            st.success("✅ Đã tìm thấy khuôn mặt!")
            st.image(face_img, caption="Khuôn mặt đã cắt (Input cho Model)", width=150)
            
            # Lưu khuôn mặt vào biến session state hoặc biến tạm để dùng ở cột bên kia
            st.session_state['face_to_process'] = face_img
        else:
            st.warning("⚠️ Không tìm thấy khuôn mặt rõ ràng. Sẽ dùng toàn bộ ảnh.")
            st.session_state['face_to_process'] = input_array

with col2:
    st.subheader("2. Kết quả dự đoán")
    
    if 'face_to_process' in st.session_state and input_image_pil is not None:
        if st.button("Chạy dự đoán", type="primary"):
            face_to_analyze = st.session_state['face_to_process']
            
            with st.spinner('Đang tính toán...'):
                try:
                    # 1. Trích xuất đặc trưng từ ảnh MẶT (chứ không phải ảnh gốc)
                    features = extract_features_from_face(face_to_analyze, selector)
                    
                    emotion_labels = {0: "Giận dữ 😡", 1: "Sợ hãi 😱", 2: "Vui vẻ 😄", 3: "Buồn 😢", 4: "Ngạc nhiên 😲"}
                    
                    # 2. Hiển thị kết quả
                    if not models:
                        st.error("Chưa tải được model.")
                    else:
                        st.write("---")
                        res_cols = st.columns(2)
                        for i, (name, model) in enumerate(models.items()):
                            pred = model.predict(features)[0]
                            label = emotion_labels.get(pred, str(pred))
                            
                            with res_cols[i % 2]:
                                st.success(f"**{name}**: {label}")
                                
                except Exception as e:
                    st.error(f"Lỗi: {e}")
    else:
        st.info("👈 Vui lòng chọn ảnh bên trái trước.")