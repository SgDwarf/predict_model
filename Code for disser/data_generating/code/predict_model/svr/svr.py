import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

# Обучение модели SVR для Prepod_mark
model_prepod_svr = SVR(kernel='rbf', C=100, gamma=0.1)
model_prepod_svr.fit(X_train, y_train_prepod)

# Прогнозирование для тестовой выборки
y_pred_prepod_svr = model_prepod_svr.predict(X_test)

# Оценка модели
mse_prepod_svr = mean_squared_error(y_test_prepod, y_pred_prepod_svr)
r2_prepod_svr = r2_score(y_test_prepod, y_pred_prepod_svr)
mae_prepod_svr = mean_absolute_error(y_test_prepod, y_pred_prepod_svr)
rmse_prepod_svr = np.sqrt(mse_prepod_svr)
mape_prepod_svr = np.mean(np.abs((y_test_prepod - y_pred_prepod_svr) / y_test_prepod)) * 100

print(f'SVR (Prepod_mark) - MSE: {mse_prepod_svr}, R2: {r2_prepod_svr}, MAE: {mae_prepod_svr}, RMSE: {rmse_prepod_svr}, MAPE: {mape_prepod_svr}')

# Обучение модели SVR для VKR
model_vkr_svr = SVR(kernel='rbf', C=100, gamma=0.1)
model_vkr_svr.fit(X_train, y_train_vkr)

# Прогнозирование для тестовой выборки
y_pred_vkr_svr = model_vkr_svr.predict(X_test)

# Оценка модели
mse_vkr_svr = mean_squared_error(y_test_vkr, y_pred_vkr_svr)
r2_vkr_svr = r2_score(y_test_vkr, y_pred_vkr_svr)
mae_vkr_svr = mean_absolute_error(y_test_vkr, y_pred_vkr_svr)
rmse_vkr_svr = np.sqrt(mse_vkr_svr)
mape_vkr_svr = np.mean(np.abs((y_test_vkr - y_pred_vkr_svr) / y_test_vkr)) * 100 if np.all(y_test_vkr != 0) else float('inf')

print(f'SVR (VKR) - MSE: {mse_vkr_svr}, R2: {r2_vkr_svr}, MAE: {mae_vkr_svr}, RMSE: {rmse_vkr_svr}, MAPE: {mape_vkr_svr}')

# Сохранение моделей
joblib.dump(model_prepod_svr, 'data_generating/code/predict_model/SMOTE_data/svr/model_prepod_svr.pkl')
joblib.dump(model_vkr_svr, 'data_generating/code/predict_model/SMOTE_data/svr/model_vkr_svr.pkl')

# Сохранение результатов
results_svr = {
    'Prepod_mark': {
        'MSE': mse_prepod_svr,
        'R2': r2_prepod_svr,
        'MAE': mae_prepod_svr,
        'RMSE': rmse_prepod_svr,
        'MAPE': mape_prepod_svr
    },
    'VKR': {
        'MSE': mse_vkr_svr,
        'R2': r2_vkr_svr,
        'MAE': mae_vkr_svr,
        'RMSE': rmse_vkr_svr,
        'MAPE': mape_vkr_svr
    }
}
results_file_path_svr = 'data_generating/code/predict_model/SMOTE_data/svr/results_svr.json'
pd.DataFrame(results_svr).to_json(results_file_path_svr, orient='index')

print(f'Results successfully saved to {results_file_path_svr}')

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

plot_results(y_test_prepod, y_pred_prepod_svr, 'Prepod_mark_SVR')
plot_results(y_test_vkr, y_pred_vkr_svr, 'VKR_SVR')

# Создание приложения Flet для отображения диаграмм
def main(page: Page):
    page.title = "Prediction Results Visualization - SVR"
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window_width = 1200
    page.window_height = 800

    page.appbar = AppBar(title=Text("Prediction Results Visualization - SVR"))

    container_prepod = Container(content=Image(src=f'Prepod_mark_SVR.png', width=1200, height=400))
    container_vkr = Container(content=Image(src=f'VKR_SVR.png', width=1200, height=400))

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
