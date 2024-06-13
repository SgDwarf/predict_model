import flet as ft
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from io import BytesIO
import base64

def main(page: ft.Page):
    page.title = "Визуализация тестирования моделей"
    page.window_width = 1200
    page.window_height = 900
    
    # Загрузка моделей и скалера
    model_prepod_path = r'data_generating\code\predict_model\SMOTE_data\svr\model_prepod_svr.pkl'
    model_vkr_path = r'data_generating\code\predict_model\SMOTE_data\svr\model_vkr_svr.pkl'
    scaler_path = 'data_generating/code/predict_model/SMOTE_data/scaler.pkl'
    
    model_prepod = joblib.load(model_prepod_path)
    model_vkr = joblib.load(model_vkr_path)
    scaler = joblib.load(scaler_path)
    
    # Загрузка и подготовка данных
    data_path = r'data_generating\files\data.csv'
    data = pd.read_csv(data_path)
    
    features = [
        'Certificate', 'Algebra', 'Geometry', 'Physics', 'Computer Science', 'Russian Language',
        'Literature', 'History', 'Biology', 'Chemistry', 'Social Studies', 'Geography',
        'Physical Education', 'Foreign Language', 'Safety Basics', 'Music', 'Art', 'Technology',
        'Physics(EGE)', 'Math(EGE)', 'Math(OGE)'
    ]
    
    target_prepod = 'Prepod_mark'
    target_vkr = 'VKR'
    
    X = data[features]
    y_prepod = data[target_prepod]
    y_vkr = data[target_vkr]
    
    # Нормализация данных
    X_normalized = scaler.transform(X)
    
    # Предсказания
    y_pred_prepod = model_prepod.predict(X_normalized)
    y_pred_vkr = model_vkr.predict(X_normalized)
    
    # Расчет метрик для Prepod_mark
    mse_prepod = mean_squared_error(y_prepod, y_pred_prepod)
    r2_prepod = r2_score(y_prepod, y_pred_prepod)
    mae_prepod = mean_absolute_error(y_prepod, y_pred_prepod)
    
    # Расчет метрик для VKR
    mse_vkr = mean_squared_error(y_vkr, y_pred_vkr)
    r2_vkr = r2_score(y_vkr, y_pred_vkr)
    mae_vkr = mean_absolute_error(y_vkr, y_pred_vkr)
    
    # Создание графика для Prepod_mark
    fig, ax = plt.subplots()
    ax.plot(y_prepod, label='Реальные данные', color='blue')
    ax.plot(y_pred_prepod, label='Спрогнозированные данные', color='red')
    ax.legend()
    ax.set_title('Prepod_mark: Реальные и спрогнозированные данные')
    
    buf_prepod = BytesIO()
    plt.savefig(buf_prepod, format='png')
    buf_prepod.seek(0)
    
    # Создание графика для VKR
    fig, ax = plt.subplots()
    ax.plot(y_vkr, label='Реальные данные', color='blue')
    ax.plot(y_pred_vkr, label='Спрогнозированные данные', color='red')
    ax.legend()
    ax.set_title('VKR: Реальные и спрогнозированные данные')
    
    buf_vkr = BytesIO()
    plt.savefig(buf_vkr, format='png')
    buf_vkr.seek(0)
    
    # Flet компоненты для отображения метрик и графиков
    metrics_column = ft.Column([
        ft.Text(f"Prepod_mark - MSE: {mse_prepod:.2f}"),
        ft.Text(f"Prepod_mark - R2: {r2_prepod:.2f}"),
        ft.Text(f"Prepod_mark - MAE: {mae_prepod:.2f}"),
        ft.Text(f"VKR - MSE: {mse_vkr:.2f}"),
        ft.Text(f"VKR - R2: {r2_vkr:.2f}"),
        ft.Text(f"VKR - MAE: {mae_vkr:.2f}")
    ], alignment=ft.MainAxisAlignment.START, expand=True)
    
    graph_image_prepod = ft.Image(src_base64=base64.b64encode(buf_prepod.read()).decode(), expand=True)
    graph_image_vkr = ft.Image(src_base64=base64.b64encode(buf_vkr.read()).decode(), expand=True)
    
    page.add(
        ft.Row(
            [
                metrics_column,
                ft.Column([graph_image_prepod, graph_image_vkr], alignment=ft.MainAxisAlignment.CENTER, expand=True)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True
        )
    )

ft.app(target=main)
