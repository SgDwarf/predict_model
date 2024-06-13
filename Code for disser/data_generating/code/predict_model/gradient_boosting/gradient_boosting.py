import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

# Обучение модели градиентного бустинга для Prepod_mark
model_prepod_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_prepod_gb.fit(X_train, y_train_prepod)

# Прогнозирование для тестовой выборки
y_pred_prepod_gb = model_prepod_gb.predict(X_test)

# Оценка модели
mse_prepod_gb = mean_squared_error(y_test_prepod, y_pred_prepod_gb)
r2_prepod_gb = r2_score(y_test_prepod, y_pred_prepod_gb)
mae_prepod_gb = mean_absolute_error(y_test_prepod, y_pred_prepod_gb)
rmse_prepod_gb = np.sqrt(mse_prepod_gb)
mape_prepod_gb = np.mean(np.abs((y_test_prepod - y_pred_prepod_gb) / y_test_prepod)) * 100

# Обучение модели градиентного бустинга для VKR
model_vkr_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_vkr_gb.fit(X_train, y_train_vkr)

# Прогнозирование для тестовой выборки
y_pred_vkr_gb = model_vkr_gb.predict(X_test)

# Оценка модели
mse_vkr_gb = mean_squared_error(y_test_vkr, y_pred_vkr_gb)
r2_vkr_gb = r2_score(y_test_vkr, y_pred_vkr_gb)
mae_vkr_gb = mean_absolute_error(y_test_vkr, y_pred_vkr_gb)
rmse_vkr_gb = np.sqrt(mse_vkr_gb)
mape_vkr_gb = np.mean(np.abs((y_test_vkr - y_pred_vkr_gb) / y_test_vkr)) * 100 if np.all(y_test_vkr != 0) else float('inf')

# Сохранение моделей
joblib.dump(model_prepod_gb, 'data_generating/code/predict_model/SMOTE_data/gradient_boosting/model_prepod_gb.pkl')
joblib.dump(model_vkr_gb, 'data_generating/code/predict_model/SMOTE_data/gradient_boosting/model_vkr_gb.pkl')

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

plot_results(y_test_prepod, y_pred_prepod_gb, 'Prepod_mark')
plot_results(y_test_vkr, y_pred_vkr_gb, 'VKR')

# Создание приложения Flet для отображения диаграмм
def main(page: Page):
    page.title = "Prediction Results Visualization"
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window_width = 1200
    page.window_height = 800

    page.appbar = AppBar(title=Text("Prediction Results Visualization"))

    container_prepod = Container(content=Image(src=f'Prepod_mark.png', width=1200, height=400))
    container_vkr = Container(content=Image(src=f'VKR.png', width=1200, height=400))

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
