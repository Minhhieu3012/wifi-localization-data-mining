# 📡 Phân Tích & Khai Thác Dữ Liệu Định Vị Trong Nhà Bằng Tín Hiệu WiFi (WiFi Localization)

Thực hiện khám phá dữ liệu (EDA), giảm chiều trực quan hóa (PCA), huấn luyện mô hình phân lớp có tinh chỉnh siêu tham số (`GridSearchCV` với 10-Fold Cross-Validation) và gom cụm không giám sát (`K-Means`, `DBSCAN`) trên bộ dữ liệu **D08 - WiFi Localization**.

---

## Tổng quan bài toán & Bộ dữ liệu

- **Bài toán:** Xác định vị trí người dùng (1 trong 4 căn phòng) trong không gian kín dựa trên cường độ tín hiệu WiFi (RSSI - đơn vị dBm) thu được từ 7 trạm phát (Access Points) trên điện thoại thông minh.
- **Nguồn dữ liệu gốc:** `data/raw/D08 - wifi_localization.xlsx` (Sheet: `wifi_localization`).
- **Quy mô thực tế:**
  - **Số mẫu:** `2.000 mẫu` (Lưu ý: file gốc không có dòng tiêu đề `header`; nếu đọc mặc định `header=0` sẽ bị mất 1 mẫu đầu tiên của Phòng 1 và còn 1.999 mẫu).
  - **Số thuộc tính đặc trưng:** `7 thuộc tính liên tục` (`WiFi_1` đến `WiFi_7`), biểu thị cường độ tín hiệu RSSI (dBm), giá trị từ `-98.0 dBm` đến `-10.0 dBm`.
  - **Nhãn phân lớp (`Room`):** `4 lớp` (`1`, `2`, `3`, `4`), phân bố cân bằng tuyệt đối (`500 mẫu/phòng`).

---

## Cấu trúc thư mục

```text
wifi-localization-data-mining/
├── data/
│   ├── raw/
│   │   └── D08 - wifi_localization.xlsx       # File Excel gốc từ đề bài
│   └── processed/
│       └── wifi_localization.csv              # File CSV đã chuẩn hóa header (2000 dòng x 8 cột)
├── notebooks/
│   ├── members/
│   │   ├── 01_eda_preprocessing.ipynb
│   │   ├── 02_pca_visualization.ipynb
│   │   ├── 03_classification_knn_svm.ipynb
│   │   ├── 04_rf_and_kmeans.ipynb
│   │   └── 05_dbscan_evaluation.ipynb
│   └── final/                                 # Sản phẩm tổng hợp
│       ├── Main_WiFi_Localization_Group.ipynb
│       └── Main_WiFi_Localization_Group.pdf
├── figures/                                   # Lưu trữ biểu đồ xuất ra từ code
├── .gitignore
├── requirements.txt
└── README.md
```
