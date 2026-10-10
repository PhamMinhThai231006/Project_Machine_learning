# XGBoost Regression – California Housing Price Prediction

## 1. Giới thiệu

Dự án sử dụng thuật toán **XGBoost Regression** để dự đoán giá trị nhà ở tại California dựa trên các đặc trưng liên quan đến thu nhập, tuổi nhà, số phòng, dân số và vị trí địa lý.

Mục tiêu của dự án là tìm hiểu quy trình xây dựng mô hình Machine Learning cho bài toán hồi quy, từ bước khám phá dữ liệu đến huấn luyện, đánh giá và trực quan hóa kết quả dự đoán.

## 2. Công nghệ sử dụng

* **Python** – ngôn ngữ lập trình.
* **Pandas** – đọc và xử lý dữ liệu.
* **Matplotlib** – trực quan hóa dữ liệu.
* **Seaborn** – xây dựng ma trận tương quan.
* **Scikit-learn** – chia tập dữ liệu và đánh giá mô hình.
* **XGBoost** – xây dựng mô hình hồi quy.

## 3. Dữ liệu

Dự án sử dụng bộ dữ liệu California Housing, gồm **20.640 mẫu dữ liệu và 9 cột**, trong đó có 8 đặc trưng đầu vào và 1 biến mục tiêu.

Các đặc trưng đầu vào bao gồm:

| Đặc trưng     | Ý nghĩa                                         |
| ------------- | ----------------------------------------------- |
| `MedInc`      | Thu nhập trung vị của hộ gia đình trong khu vực |
| `HouseAge`    | Tuổi nhà trung vị                               |
| `AveRooms`    | Số phòng trung bình                             |
| `AveBedrms`   | Số phòng ngủ trung bình                         |
| `Population`  | Dân số khu vực                                  |
| `AveOccup`    | Số người trung bình trong mỗi hộ                |
| `Latitude`    | Vĩ độ                                           |
| `Longitude`   | Kinh độ                                         |
| `MedHouseVal` | Giá trị nhà trung vị – biến mục tiêu            |

Biến `MedHouseVal` được biểu diễn theo đơn vị 100.000 USD.

## 4. Quy trình thực hiện

### Bước 1: Khám phá dữ liệu (EDA)

* Kiểm tra kích thước và kiểu dữ liệu.
* Kiểm tra giá trị thiếu.
* Thống kê mô tả các đặc trưng.
* Trực quan hóa phân phối dữ liệu.
* Phân tích ma trận tương quan giữa các biến.

### Bước 2: Tiền xử lý và chia dữ liệu

Tách dữ liệu thành:

* **80% dữ liệu huấn luyện (Training Set)**.
* **20% dữ liệu kiểm tra (Testing Set)**.

Sử dụng `random_state=42` để kết quả chia dữ liệu có thể tái lập.

### Bước 3: Xây dựng mô hình

Sử dụng XGBoost Regression với các tham số:

| Tham số         | Giá trị |
| --------------- | ------: |
| `n_estimators`  |     100 |
| `learning_rate` |    0.05 |
| `max_depth`     |       5 |
| `random_state`  |      42 |

Mô hình được huấn luyện trên tập dữ liệu training và dự đoán giá trị nhà trên tập testing.

### Bước 4: Đánh giá mô hình

Sử dụng các chỉ số đánh giá hồi quy:

* **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình giữa giá trị thực tế và giá trị dự đoán.
* **MSE (Mean Squared Error):** Trung bình bình phương sai số dự đoán.
* **RMSE (Root Mean Squared Error):** Căn bậc hai của MSE, thể hiện sai số theo cùng đơn vị với biến mục tiêu.
* **R² (Coefficient of Determination):** Đánh giá mức độ mô hình giải thích sự biến thiên của biến mục tiêu.

## 5. Kết quả thực nghiệm

Kết quả đánh giá mô hình trên tập testing:

| Chỉ số | Kết quả |
| ------ | ------: |
| MAE    |  0.3534 |
| MSE    |  0.2720 |
| RMSE   |  0.5215 |
| R²     |  0.7924 |

Với R² = 0.7924, mô hình giải thích được khoảng 79,24% mức biến thiên của giá trị nhà trong tập kiểm tra.

MAE = 0.3534 tương ứng với sai số tuyệt đối trung bình khoảng 35.340 USD theo đơn vị của biến mục tiêu.

## 6. Trực quan hóa kết quả

Dự án sử dụng hai biểu đồ chính:

* **Actual vs Predicted:** So sánh giá trị nhà thực tế với giá trị mô hình dự đoán. Các điểm càng gần đường tham chiếu \(y=x\) thì dự đoán càng sát giá trị thực tế.
* **Feature Importance:** Thể hiện mức độ quan trọng tương đối của các đặc trưng trong mô hình XGBoost.

Các biểu đồ hỗ trợ đánh giá hiệu quả dự đoán và tìm hiểu những đặc trưng được mô hình sử dụng nhiều trong quá trình dự đoán.

## 7. Nhận xét và kết luận

Mô hình XGBoost Regression đạt R² = 0.7924 trên tập kiểm tra, cho thấy khả năng giải thích tương đối tốt sự biến thiên của giá trị nhà trong bộ dữ liệu. Tuy nhiên, mô hình vẫn có sai số dự đoán, thể hiện qua các chỉ số MAE và RMSE.

Thông qua dự án, nhóm đã thực hiện được quy trình cơ bản của một bài toán hồi quy trong Machine Learning, bao gồm khám phá dữ liệu, chia tập huấn luyện và kiểm tra, huấn luyện mô hình, đánh giá hiệu suất và trực quan hóa kết quả.

Trong tương lai, có thể thử nghiệm thêm các bộ tham số, sử dụng cross-validation và tối ưu hóa siêu tham số để tìm mô hình phù hợp hơn.

## 8. Cấu trúc thư mục

```text
XG_boost/
├── README.md
└── xgboost_regression.ipynb
```

File dữ liệu `california_housing.csv` cần được đặt tại vị trí phù hợp với đường dẫn đọc dữ liệu trong notebook.

## 9. Hướng dẫn chạy chương trình

### Yêu cầu

* Python 3
* Jupyter Notebook hoặc JupyterLab

### Cài đặt thư viện

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Chạy notebook

1. Tải hoặc clone repository về máy.
2. Mở terminal tại thư mục dự án.
3. Khởi động Jupyter Notebook:

```bash
jupyter notebook
```

4. Mở file `xgboost_regression.ipynb`.
5. Đảm bảo file `california_housing.csv` nằm đúng vị trí mà notebook yêu cầu.
6. Chạy các cell theo thứ tự từ trên xuống.

## 10. Tác giả

Dự án được thực hiện trong khuôn khổ bài tập Machine Learning, tập trung vào thuật toán XGBoost Regression và bài toán dự đoán giá trị nhà tại California.
