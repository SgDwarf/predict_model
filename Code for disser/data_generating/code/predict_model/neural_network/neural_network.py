import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# Построение модели нейронной сети для Prepod_mark
model_prepod_nn = Sequential()
model_prepod_nn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_prepod_nn.add(Dense(32, activation='relu'))
model_prepod_nn.add(Dense(1))

model_prepod_nn.compile(loss='mean_squared_error', optimizer='adam')
model_prepod_nn.fit(X_train, y_train_prepod, epochs=50, batch_size=10, verbose=1)

# Прогнозирование для тестовой выборки
y_pred_prepod_nn = model_prepod_nn.predict(X_test).flatten()

# Оценка модели
mse_prepod_nn = mean_squared_error(y_test_prepod, y_pred_prepod_nn)
r2_prepod_nn = r2_score(y_test_prepod, y_pred_prepod_nn)
print(f'Neural Network (Prepod_mark) - MSE: {mse_prepod_nn}, R2: {r2_prepod_nn}')

# Построение модели нейронной сети для VKR
model_vkr_nn = Sequential()
model_vkr_nn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_vkr_nn.add(Dense(32, activation='relu'))
model_vkr_nn.add(Dense(1))

model_vkr_nn.compile(loss='mean_squared_error', optimizer='adam')
model_vkr_nn.fit(X_train, y_train_vkr, epochs=50, batch_size=10, verbose=1)

# Прогнозирование для тестовой выборки
y_pred_vkr_nn = model_vkr_nn.predict(X_test).flatten()

# Оценка модели
mse_vkr_nn = mean_squared_error(y_test_vkr, y_pred_vkr_nn)
r2_vkr_nn = r2_score(y_test_vkr, y_pred_vkr_nn)
print(f'Neural Network (VKR) - MSE: {mse_vkr_nn}, R2: {r2_vkr_nn}')

# Сохранение моделей
model_prepod_nn.save('data_generating/code/predict_model/SMOTE_data/neural_network/model_prepod_nn.keras')
model_vkr_nn.save('data_generating/code/predict_model/SMOTE_data/neural_network/model_vkr_nn.keras')

# Сохранение результатов
results_nn = {
    'Prepod_mark': {'MSE': mse_prepod_nn, 'R2': r2_prepod_nn},
    'VKR': {'MSE': mse_vkr_nn, 'R2': r2_vkr_nn}
}
results_file_path_nn = 'data_generating/code/predict_model/SMOTE_data/neural_network/results_nn.json'
pd.DataFrame(results_nn).to_json(results_file_path_nn, orient='index')

print(f'Results successfully saved to {results_file_path_nn}')

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

plot_results(y_test_prepod, y_pred_prepod_nn, 'Prepod_mark_NeuralNetwork')
plot_results(y_test_vkr, y_pred_vkr_nn, 'VKR_NeuralNetwork')

# Создание приложения Flet для отображения диаграмм
def main(page: Page):
    page.title = "Prediction Results Visualization - Neural Network"
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window_width = 1200
    page.window_height = 800

    page.appbar = AppBar(title=Text("Prediction Results Visualization - Neural Network"))

    container_prepod = Container(content=Image(src=f'Prepod_mark_NeuralNetwork.png', width=1200, height=400))
    container_vkr = Container(content=Image(src=f'VKR_NeuralNetwork.png', width=1200, height=400))

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
