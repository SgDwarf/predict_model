import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

# Обучение модели Лассо-регрессии для Prepod_mark
model_prepod_lasso = Lasso(alpha=0.1)
model_prepod_lasso.fit(X_train, y_train_prepod)

# Прогнозирование для тестовой выборки
y_pred_prepod_lasso = model_prepod_lasso.predict(X_test)

# Оценка модели
mse_prepod_lasso = mean_squared_error(y_test_prepod, y_pred_prepod_lasso)
r2_prepod_lasso = r2_score(y_test_prepod, y_pred_prepod_lasso)
mae_prepod_lasso = mean_absolute_error(y_test_prepod, y_pred_prepod_lasso)
rmse_prepod_lasso = np.sqrt(mse_prepod_lasso)
mape_prepod_lasso = np.mean(np.abs((y_test_prepod - y_pred_prepod_lasso) / y_test_prepod)) * 100

print(f'Lasso Regression (Prepod_mark) - MSE: {mse_prepod_lasso}, R2: {r2_prepod_lasso}, MAE: {mae_prepod_lasso}, RMSE: {rmse_prepod_lasso}, MAPE: {mape_prepod_lasso}')

# Обучение модели Лассо-регрессии для VKR
model_vkr_lasso = Lasso(alpha=0.1)
model_vkr_lasso.fit(X_train, y_train_vkr)

# Прогнозирование для тестовой выборки
y_pred_vkr_lasso = model_vkr_lasso.predict(X_test)

# Оценка модели
mse_vkr_lasso = mean_squared_error(y_test_vkr, y_pred_vkr_lasso)
r2_vkr_lasso = r2_score(y_test_vkr, y_pred_vkr_lasso)
mae_vkr_lasso = mean_absolute_error(y_test_vkr, y_pred_vkr_lasso)
rmse_vkr_lasso = np.sqrt(mse_vkr_lasso)
mape_vkr_lasso = np.mean(np.abs((y_test_vkr - y_pred_vkr_lasso) / y_test_vkr)) * 100 if np.all(y_test_vkr != 0) else float('inf')

print(f'Lasso Regression (VKR) - MSE: {mse_vkr_lasso}, R2: {r2_vkr_lasso}, MAE: {mae_vkr_lasso}, RMSE: {rmse_vkr_lasso}, MAPE: {mape_vkr_lasso}')

# Сохранение моделей
joblib.dump(model_prepod_lasso, 'data_generating/code/predict_model/SMOTE_data/lasso_regression/model_prepod_lasso.pkl')
joblib.dump(model_vkr_lasso, 'data_generating/code/predict_model/SMOTE_data/lasso_regression/model_vkr_lasso.pkl')

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

plot_results(y_test_prepod, y_pred_prepod_lasso, 'Prepod_mark_Lasso')
plot_results(y_test_vkr, y_pred_vkr_lasso, 'VKR_Lasso')

# Создание приложения Flet для отображения диаграмм
def main(page: Page):
    page.title = "Prediction Results Visualization - Lasso Regression"
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window_width = 1200
    page.window_height = 800

    page.appbar = AppBar(title=Text("Prediction Results Visualization - Lasso Regression"))

    container_prepod = Container(content=Image(src=f'Prepod_mark_Lasso.png', width=1200, height=400))
    container_vkr = Container(content=Image(src=f'VKR_Lasso.png', width=1200, height=400))

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
