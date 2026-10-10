import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

#Tai mo hinh da huan luyen
model = joblib.load("random_forest.pkl")

#Doc du lieu
df = pd.read_csv(r"D:\py\ML\california_housing.csv", sep=",")

#Tach dac trung va gia tri thuc te
x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

#Chia lai du lieu giong train.py
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Lay dong dau tien cua tap test
sample = x_test.iloc[[0]]
actual = y_test.iloc[0]

#Du doan
prediction = model.predict(sample)

print("Gia tri du doan:", prediction[0])
print("Gia tri thuc te:", actual)
print("Sai so tuyet doi:", abs(prediction[0] - actual))
