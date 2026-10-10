import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

#Doc du lieu
df = pd.read_csv(r"D:\py\ML\california_housing.csv", sep=",")

#Chia x va y
x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

#Train va test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("X:", x.shape)
print("y:", y.shape)
print("X_train:", x_train.shape)
print("X_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

#Thu nghiem cac gia tri max_depth
result = []
for depth in [None, 10, 20]:
    model = RandomForestRegressor(
        n_estimators = 200,
        max_depth = depth,
        random_state = 42,
        n_jobs = -1
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Max_depth: ", depth)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    result.append({
        "Max_depth": depth,
        "Mae": mae,
        "Mse": mse,
        "R2": r2
    })

#So sanh ket qua
result_df = pd.DataFrame(result)
print("Bang so sanh")
print(result_df)

#Chon mo hinh R2 cao nhat
best_depth = result_df.loc[result_df["R2"].idxmax(), "Max_depth"]

if pd.isna(best_depth):
    best_depth = None
else:
    best_depth = int(best_depth)

print("Max depth tot nhat trong thu nghiem:", best_depth)

#Huan luyen lai mo hinh moi dc chon
best_model = RandomForestRegressor(
    n_estimators = 200,
    max_depth = best_depth,
    random_state = 42,
    n_jobs = -1
)

best_model.fit(x_train, y_train)

#Luu
joblib.dump(best_model, "random_forest.pkl")
print("ok")