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
    model_prepod_path = r'data_generating\code\predict_model\SMOTE_data\random_forest\model_prepod_rf.pkl'
    model_vkr_path = r'data_generating\code\predict_model\SMOTE_data\random_forest\model_vkr_rf.pkl'
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
    
    X = data[features].astype(float)
    y_prepod = data[target_prepod]
    y_vkr = data[target_vkr]
    
    # Нормализация данных
    X_normalized = scaler.transform(X)
    
    # Предсказания
    y_pred_prepod = model_prepod.predict(X_normalized)
    y_pred_vkr = model_vkr.predict(X_normalized)
    
    # Создание столбчатой диаграммы для Prepod_mark
    indices = np.arange(len(y_prepod))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(indices, y_prepod, width, label='Реальные данные', color='blue')
    ax.bar(indices + width, y_pred_prepod, width, label='Спрогнозированные данные', color='red')
    ax.legend()
    ax.set_title('Prepod_mark: Реальные и спрогнозированные данные')
    
    buf_prepod = BytesIO()
    plt.savefig(buf_prepod, format='png', bbox_inches='tight')
    buf_prepod.seek(0)
    
    # Создание столбчатой диаграммы для VKR
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(indices, y_vkr, width, label='Реальные данные', color='blue')
    ax.bar(indices + width, y_pred_vkr, width, label='Спрогнозированные данные', color='red')
    ax.legend()
    ax.set_title('VKR: Реальные и спрогнозированные данные')
    
    buf_vkr = BytesIO()
    plt.savefig(buf_vkr, format='png', bbox_inches='tight')
    buf_vkr.seek(0)
    
    # Flet компоненты для отображения графиков
    graph_image_prepod = ft.Image(src_base64=base64.b64encode(buf_prepod.read()).decode(), width=1200, height=450, fit=ft.ImageFit.CONTAIN)
    graph_image_vkr = ft.Image(src_base64=base64.b64encode(buf_vkr.read()).decode(), width=1200, height=450, fit=ft.ImageFit.CONTAIN)
    
    page.add(
        ft.Column(
            [
                graph_image_prepod,
                graph_image_vkr
            ],
            alignment=ft.MainAxisAlignment.START,
            expand=True
        )
    )

ft.app(target=main)
