# Decision Tree Regression

## 1. Giới thiệu

Phần này xây dựng và đánh giá mô hình Decision Tree Regression để dự đoán giá trị trung vị nhà ở California. Dữ liệu được đọc từ `../data/california_housing.csv`, gồm 20.640 mẫu, 8 đặc trưng đầu vào (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`) và biến mục tiêu `MedHouseVal`. Dữ liệu không có giá trị thiếu; đơn vị mục tiêu là 100.000 USD.

## 2. Cách chạy

Cài đặt các thư viện cần thiết từ thư mục `tree_decision/`:

```bash
pip install -r requirements.txt
```

Mở `decision_tree_regression.ipynb` trong Jupyter hoặc VS Code và chạy lần lượt từ đầu đến cuối. Notebook tạo các biểu đồ tại `figures/eda.png`, `figures/result.png` và `figures/tree.png`.

## 3. Quy trình mô hình

Tập dữ liệu được chia train/test theo tỉ lệ 80/20 với `random_state=42`. Mô hình baseline là `DecisionTreeRegressor(random_state=42)` không giới hạn độ sâu. GridSearchCV dùng 5-fold cross-validation, tối ưu `neg_mean_squared_error` trên 16 tổ hợp: `max_depth` ∈ {5, 8, 10, 15} và `min_samples_leaf` ∈ {10, 20, 30, 50}.

Tham số tốt nhất là `max_depth=15`, `min_samples_leaf=20`. Bảng dưới đây so sánh MAE, RMSE và R² trên train/test; MAE và RMSE tính theo đơn vị 100.000 USD, R² không có đơn vị.

| Mô hình | Tập dữ liệu | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Baseline | Train | 0.0000 | 0.0000 | 1.0000 |
| Baseline | Test | 0.4542 | 0.7030 | 0.6228 |
| Tuned | Train | 0.3384 | 0.5055 | 0.8088 |
| Tuned | Test | 0.4041 | 0.5979 | 0.7272 |

## 4. Nhận xét

- Baseline khớp hoàn hảo trên train nhưng chỉ đạt R² = 0.6228 trên test, cho thấy overfitting rõ rệt.
- Sau khi giới hạn độ sâu và tăng số mẫu tối thiểu ở lá, khoảng cách R² train-test giảm từ khoảng 0.3772 xuống 0.0816; R² test tăng lên 0.7272, còn MAE/RMSE test giảm.
- `MedInc` (thu nhập trung vị) là đặc trưng quan trọng nhất trong mô hình đã tune, vượt trội so với các đặc trưng còn lại.
- `MedHouseVal` bị chặn ở mức khoảng 5.0 (500.000 USD), nên mô hình không thể học đầy đủ mức giá thực của các căn nhà có giá trị cao hơn giới hạn này.
- Biểu đồ phân phối/tương quan và biểu đồ kết quả được lưu lần lượt tại `figures/eda.png` và `figures/result.png`.
- Hình `figures/tree.png` chỉ minh họa 2 tầng đầu của cây; mô hình thực tế có `max_depth=15` nên sâu hơn nhiều.
