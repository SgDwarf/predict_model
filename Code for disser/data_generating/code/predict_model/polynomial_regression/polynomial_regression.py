import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import matplotlib.pyplot as plt
from flet import Page, Column, Container, Row, Text, Image, AppBar, ScrollMode

# Загрузка нормализованных данных
file_path = 'data_generating/code/predict_model/SMOTE_data/normalized_data.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# Определение признаков и целевых переменных
features = [
    'Certificate', 'Algebra', 'Geometry', 'Physics', 'Computer Science', 'Russian Language',
    'Literature', 'History', 'Biology', 'Chemistry', 'Social Studies', 'Geography',
    'Physical Education', 'Foreign Language', 'Safety Basics', 'Music', 'Art', 'Technology',
    'Physics(EGE)', 'Math(EGE)', 'Math(OGE)'
]
target_prepod = 'Prepod_mark'
target_vkr = 'VKR'

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train_prepod, y_test_prepod = train_test_split(data[features], data[target_prepod], test_size=0.2, random_state=42)
_, _, y_train_vkr, y_test_vkr = train_test_split(data[features], data[target_vkr], test_size=0.2, random_state=42)

# Создание полиномиальных признаков
degree = 2  # Степень полинома
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Обучение модели полиномиальной регрессии с регуляризацией (Ridge) для Prepod_mark
model_prepod_poly_ridge = Ridge(alpha=1.0)
model_prepod_poly_ridge.fit(X_train_poly, y_train_prepod)

# Прогнозирование для тестовой выборки
y_pred_prepod_poly_ridge = model_prepod_poly_ridge.predict(X_test_poly)

# Оценка модели
mse_prepod_poly_ridge = mean_squared_error(y_test_prepod, y_pred_prepod_poly_ridge)
r2_prepod_poly_ridge = r2_score(y_test_prepod, y_pred_prepod_poly_ridge)
mae_prepod_poly_ridge = mean_absolute_error(y_test_prepod, y_pred_prepod_poly_ridge)
rmse_prepod_poly_ridge = np.sqrt(mse_prepod_poly_ridge)
mape_prepod_poly_ridge = np.mean(np.abs((y_test_prepod - y_pred_prepod_poly_ridge) / y_test_prepod)) * 100

print(f'Polynomial Regression with Ridge (Prepod_mark) - MSE: {mse_prepod_poly_ridge}, R2: {r2_prepod_poly_ridge}, MAE: {mae_prepod_poly_ridge}, RMSE: {rmse_prepod_poly_ridge}, MAPE: {mape_prepod_poly_ridge}')

# Обучение модели полиномиальной регрессии с регуляризацией (Ridge) для VKR
model_vkr_poly_ridge = Ridge(alpha=1.0)
model_vkr_poly_ridge.fit(X_train_poly, y_train_vkr)

# Прогнозирование для тестовой выборки
y_pred_vkr_poly_ridge = model_vkr_poly_ridge.predict(X_test_poly)

# Оценка модели
mse_vkr_poly_ridge = mean_squared_error(y_test_vkr, y_pred_vkr_poly_ridge)
r2_vkr_poly_ridge = r2_score(y_test_vkr, y_pred_vkr_poly_ridge)
mae_vkr_poly_ridge = mean_absolute_error(y_test_vkr, y_pred_vkr_poly_ridge)
rmse_vkr_poly_ridge = np.sqrt(mse_vkr_poly_ridge)
mape_vkr_poly_ridge = np.mean(np.abs((y_test_vkr - y_pred_vkr_poly_ridge) / y_test_vkr)) * 100 if np.all(y_test_vkr != 0) else float('inf')

print(f'Polynomial Regression with Ridge (VKR) - MSE: {mse_vkr_poly_ridge}, R2: {r2_vkr_poly_ridge}, MAE: {mae_vkr_poly_ridge}, RMSE: {rmse_vkr_poly_ridge}, MAPE: {mape_vkr_poly_ridge}')

# Сохранение моделей
joblib.dump(model_prepod_poly_ridge, 'data_generating/code/predict_model/SMOTE_data/polynomial_regression/model_prepod_poly_ridge.pkl')
joblib.dump(model_vkr_poly_ridge, 'data_generating/code/predict_model/SMOTE_data/polynomial_regression/model_vkr_poly_ridge.pkl')

# Сохранение результатов
results_poly_ridge = {
    'Prepod_mark': {
        'MSE': mse_prepod_poly_ridge,
        'R2': r2_prepod_poly_ridge,
        'MAE': mae_prepod_poly_ridge,
        'RMSE': rmse_prepod_poly_ridge,
        'MAPE': mape_prepod_poly_ridge
    },
    'VKR': {
        'MSE': mse_vkr_poly_ridge,
        'R2': r2_vkr_poly_ridge,
        'MAE': mae_vkr_poly_ridge,
        'RMSE': rmse_vkr_poly_ridge,
        'MAPE': mape_vkr_poly_ridge
    }
}
results_file_path_poly_ridge = r'data_generating/code/predict_model/SMOTE_data/polynomial_regression/results_poly_ridge.json'
pd.DataFrame(results_poly_ridge).to_json(results_file_path_poly_ridge, orient='index')

print(f'Results successfully saved to {results_file_path_poly_ridge}')

# Визуализация результатов
def plot_results(real, predicted, title):
    plt.figure(figsize=(12, 6))
    real = real[:50]
    predicted = predicted[:50]
    indices = np.arange(len(real))
    width = 0.35
    
    plt.bar(indices, real, width, label='Real')
    plt.bar(indices + width, predicted, width, label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png')
    plt.close()

plot_results(y_test_prepod, y_pred_prepod_poly_ridge, 'Prepod_mark_PolynomialRidge')
plot_results(y_test_vkr, y_pred_vkr_poly_ridge, 'VKR_PolynomialRidge')

# Создание приложения Flet для отображения диаграмм
def main(page: Page):
    page.title = "Prediction Results Visualization - Polynomial Ridge Regression"
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window_width = 1200
    page.window_height = 800

    page.appbar = AppBar(title=Text("Prediction Results Visualization - Polynomial Ridge Regression"))

    container_prepod = Container(content=Image(src=f'Prepod_mark_PolynomialRidge.png', width=1200, height=400))
    container_vkr = Container(content=Image(src=f'VKR_PolynomialRidge.png', width=1200, height=400))

    page.add(
        Column(
            controls=[
                container_prepod,
                container_vkr
            ],
            alignment='center',
            expand=True,
            scroll=ScrollMode.AUTO
        )
    )

if __name__ == "__main__":
    from flet import app
    app(target=main)
