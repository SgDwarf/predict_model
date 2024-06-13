import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
X_train_vkr, X_test_vkr, y_train_vkr, y_test_vkr = train_test_split(data[features], data[target_vkr], test_size=0.2, random_state=42)

# Обучение модели случайного леса для Prepod_mark
model_prepod_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_prepod_rf.fit(X_train, y_train_prepod)

# Прогнозирование для тестовой выборки
y_pred_prepod_rf = model_prepod_rf.predict(X_test)

# Оценка модели
mse_prepod_rf = mean_squared_error(y_test_prepod, y_pred_prepod_rf)
r2_prepod_rf = r2_score(y_test_prepod, y_pred_prepod_rf)
mae_prepod_rf = mean_absolute_error(y_test_prepod, y_pred_prepod_rf)
rmse_prepod_rf = np.sqrt(mse_prepod_rf)
mape_prepod_rf = np.mean(np.abs((y_test_prepod - y_pred_prepod_rf) / y_test_prepod)) * 100

print(f'Random Forest (Prepod_mark) - MSE: {mse_prepod_rf}, R2: {r2_prepod_rf}, MAE: {mae_prepod_rf}, RMSE: {rmse_prepod_rf}, MAPE: {mape_prepod_rf}')

# Обучение модели случайного леса для VKR
model_vkr_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_vkr_rf.fit(X_train_vkr, y_train_vkr)

# Прогнозирование для тестовой выборки
y_pred_vkr_rf = model_vkr_rf.predict(X_test_vkr)

# Оценка модели
mse_vkr_rf = mean_squared_error(y_test_vkr, y_pred_vkr_rf)
r2_vkr_rf = r2_score(y_test_vkr, y_pred_vkr_rf)
mae_vkr_rf = mean_absolute_error(y_test_vkr, y_pred_vkr_rf)
rmse_vkr_rf = np.sqrt(mse_vkr_rf)
mape_vkr_rf = np.mean(np.abs((y_test_vkr - y_pred_vkr_rf) / y_test_vkr)) * 100 if np.all(y_test_vkr != 0) else float('inf')

print(f'Random Forest (VKR) - MSE: {mse_vkr_rf}, R2: {r2_vkr_rf}, MAE: {mae_vkr_rf}, RMSE: {rmse_vkr_rf}, MAPE: {mape_vkr_rf}')

# Сохранение моделей
joblib.dump(model_prepod_rf, 'data_generating/code/predict_model/SMOTE_data/random_forest/model_prepod_rf.pkl')
joblib.dump(model_vkr_rf, 'data_generating/code/predict_model/SMOTE_data/random_forest/model_vkr_rf.pkl')

# Сохранение результатов
results_rf = {
    'Prepod_mark': {
        'MSE': mse_prepod_rf,
        'R2': r2_prepod_rf,
        'MAE': mae_prepod_rf,
        'RMSE': rmse_prepod_rf,
        'MAPE': mape_prepod_rf
    },
    'VKR': {
        'MSE': mse_vkr_rf,
        'R2': r2_vkr_rf,
        'MAE': mae_vkr_rf,
        'RMSE': rmse_vkr_rf,
        'MAPE': mape_vkr_rf
    }
}
results_file_path_rf = 'data_generating/code/predict_model/SMOTE_data/random_forest/results_rf.json'
pd.DataFrame(results_rf).to_json(results_file_path_rf, orient='index')

print(f'Results successfully saved to {results_file_path_rf}')

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

plot_results(y_test_prepod, y_pred_prepod_rf, 'Prepod_mark_RandomForest')
plot_results(y_test_vkr, y_pred_vkr_rf, 'VKR_RandomForest')

# Создание приложения Flet для отображения диаграмм
def main(page: Page):
    page.title = "Prediction Results Visualization - Random Forest"
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window_width = 1200
    page.window_height = 800

    page.appbar = AppBar(title=Text("Prediction Results Visualization - Random Forest"))

    container_prepod = Container(content=Image(src=f'Prepod_mark_RandomForest.png', width=1200, height=400))
    container_vkr = Container(content=Image(src=f'VKR_RandomForest.png', width=1200, height=400))

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
