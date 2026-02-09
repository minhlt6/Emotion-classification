# 🎭 Facial Expression Recognition App (HOG + ML)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Completed-success)

Ứng dụng web nhận diện cảm xúc khuôn mặt sử dụng kỹ thuật trích xuất đặc trưng **HOG (Histogram of Oriented Gradients)** kết hợp với các thuật toán Machine Learning cổ điển. Hệ thống được tối ưu hóa để hoạt động tốt trên cả ảnh tĩnh và Webcam thời gian thực.

## 📖 Giới thiệu

Dự án này xây dựng hệ thống phân loại cảm xúc thành 5 nhóm cơ bản:
1. **Giận dữ (Angry)**
2. **Sợ hãi (Fear)**
3. **Vui vẻ (Happy)**
4. **Buồn (Sad)**
5. **Ngạc nhiên (Surprise)**

Hệ thống so sánh hiệu quả của 4 thuật toán phân loại khác nhau trên cùng một tập dữ liệu:
- **Random Forest (RF)**
- **K-Nearest Neighbors (KNN)**
- **ID3 (Decision Tree)**
- **CART (Decision Tree)**

## 🚀 Tính năng nổi bật

- **📸 Đa dạng đầu vào:** Hỗ trợ tải ảnh lên (Upload) hoặc chụp trực tiếp từ Webcam.
- **🤖 Tự động phát hiện khuôn mặt:** Sử dụng **Haar Cascade** để định vị và cắt khuôn mặt chính xác (Tight Crop) loại bỏ nhiễu nền.
- **⚙️ Xử lý ảnh nâng cao (Preprocessing):**
  - Resize chuẩn **64x64**.
  - **Cân bằng sáng (Histogram Equalization):** Giúp nhận diện tốt trong điều kiện thiếu sáng.
  - **Khử nhiễu (Gaussian Blur):** Loại bỏ nhiễu hạt từ camera.
- **📉 Giảm chiều dữ liệu:** Tích hợp bước **Feature Selection** (loại bỏ đặc trưng có phương sai = 0) giúp mô hình nhẹ và nhanh hơn.
- **📊 So sánh trực quan:** Hiển thị kết quả dự đoán của cả 4 thuật toán cùng lúc.

## 🛠️ Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **Python** | Ngôn ngữ lập trình chính |
| **Streamlit** | Xây dựng giao diện web (Web App) |
| **OpenCV** | Xử lý ảnh, phát hiện khuôn mặt, tính HOG |
| **Scikit-learn** | Huấn luyện mô hình, giảm chiều dữ liệu |
| **Joblib** | Lưu trữ và tải mô hình (.pkl) |
| **Gdown** | Tải model tự động từ Google Drive |

## ⚙️ Cài đặt và Sử dụng

### 1. Clone dự án về máy
```bash
git clone [https://github.com/username/ten-du-an-cua-ban.git](https://github.com/username/ten-du-an-cua-ban.git)
cd ten-du-an-cua-ban
```
### 2. Cài đặt các thư viện cần thiết 
Hãy đảm bảo bạn đã cài Python và cài đặt các thư viện phụ thuộc :
```bash
pip install -r requirement.text
```
### 3.Các file model và selector 
Các file này được hỗ trợ để tải xuống từ Google Drive 

### 4. Chạy ứng dụng 
Mở Terminal và chạy dòng lệnh :
```bash
streamlist run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: (http://localhost:8501)
## 📂 Cấu trúc thư mục
```text
├── app.py                 # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt       # Danh sách các thư viện cần cài đặt
├── README.md              # Tài liệu hướng dẫn sử dụng
├── selector.pkl           # File giảm chiều dữ liệu (VarianceThreshold)
├── rf_model.pkl           # Model Random Forest
├── knn_model.pkl          # Model KNN
├── cart_model.pkl         # Model CART
└── id3_model.pkl          # Model ID3
```
## 📈 Quy trình xử lý (Pipeline)
Để đảm bảo tính đồng nhất trong quá trình huấn luyện và dự đoán , phải tuân thủ theo các bước :
1. **Input Image (Webcam/Upload)**
2. **Face Detection (Haar Cascade) -> Lấy tọa độ**
3. **Tight Crop (Cắt bỏ viền tóc, cổ để tập trung vào cơ mặt)**
4. **Resize (Về kích thước 64x64 pixel)**
5. **Histogram Equalization (Tăng độ tương phản)**
6. **HOG Feature Extraction (Trích xuất đặc trưng hình dạng)**
7. **Feature Selection (Dùng selector.pkl để lọc đặc trưng)**
8. **Prediction (Đưa vào 4 Model để dự đoán**
## Tác giả 
### Lê Tiến Minh
