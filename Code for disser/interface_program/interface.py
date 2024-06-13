import flet as ft
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(page: ft.Page):
    page.title = "Прогнозирование успеваемости"
    page.window_width = 1200
    page.window_height = 900

    subjects_to_average = [
        "Algebra", "Geometry", "Physics", "Computer Science", "Russian Language",
        "Literature", "History", "Biology", "Chemistry", "Social Studies", 
        "Geography", "Physical Education", "Foreign Language", "Safety Basics", 
        "Music", "Art", "Technology", "Math(OGE)"
    ]

    subjects_left = [
        "Algebra", "Geometry", "Physics", "Computer Science", "Russian Language",
        "Literature", "History", "Biology", "Chemistry", "Math(OGE)", "Physics(EGE)"
    ]
    
    subjects_right = [
        "Social Studies", "Geography", "Physical Education", "Foreign Language", 
        "Safety Basics", "Music", "Art", "Technology", "Certificate", 
        "Math(EGE)"
    ]

    fields_left = {subject: ft.TextField(label=subject, keyboard_type=ft.KeyboardType.NUMBER, height=40) for subject in subjects_left}
    fields_right = {subject: ft.TextField(label=subject, keyboard_type=ft.KeyboardType.NUMBER, height=40) for subject in subjects_right}

    average_field = fields_right["Certificate"]
    average_field.read_only = True

    for subject, field in fields_left.items():
        field.on_change = lambda e, subj=subject: validate_and_update(e, subj, fields_left, fields_right, subjects_to_average, page)
    
    for subject, field in fields_right.items():
        field.on_change = lambda e, subj=subject: validate_and_update(e, subj, fields_left, fields_right, subjects_to_average, page)

    submit_button = ft.ElevatedButton(
        text="Узнать результат",
        on_click=lambda e: show_results(page, fields_left, fields_right, subjects_to_average)
    )

    page.add(
        ft.Column(
            [
                ft.Container(
                    content=ft.Text("ПРОГНОЗИРОВАНИЕ УСПЕВАЕМОСТИ", size=24, weight="bold"),
                    alignment=ft.alignment.center,
                    padding=ft.Padding(left=0, top=10, right=0, bottom=0)
                ),
                ft.Row(
                    [
                        ft.Column(list(fields_left.values()), expand=True),
                        ft.Column(list(fields_right.values()), expand=True)
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    expand=True
                ),
                ft.Container(
                    content=submit_button,
                    alignment=ft.alignment.center,
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            expand=True
        )
    )

def results_page(page, prediction_prepod, prediction_vkr, accuracy_prepod, accuracy_vkr, recommendations):
    prediction_prepod_text = ft.Text(f"Спрогнозированное значение для преподавателя: {prediction_prepod}")
    prediction_vkr_text = ft.Text(f"Спрогнозированное значение для ВКР: {prediction_vkr}")
    accuracy_prepod_text = ft.Text(f"Точность прогноза для преподавателя: {accuracy_prepod}%")
    accuracy_vkr_text = ft.Text(f"Точность прогноза для ВКР: {accuracy_vkr}%")
    recommendations_text = ft.Text(f"Рекомендации: {recommendations}")
    
    back_button = ft.ElevatedButton(text="Назад", on_click=lambda e: go_back_to_main(page))
    
    page.clean()
    page.add(
        ft.Container(
            content=ft.Column(
                [
                    prediction_prepod_text,
                    prediction_vkr_text,
                    accuracy_prepod_text,
                    accuracy_vkr_text,
                    recommendations_text,
                    back_button
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                expand=True
            ),
            alignment=ft.alignment.center,
        )
    )

def go_back_to_main(page):
    page.clean()
    main(page)

def validate_and_update(e, subject, fields_left, fields_right, subjects_to_average, page):
    if subject in subjects_to_average:
        validate_grade(e)
        update_average(fields_left, fields_right, subjects_to_average, page)
    elif subject in ["Math(OGE)", "Physics(EGE)", "Math(EGE)"]:
        validate_oge_ege(e)
    else:
        validate_integer(e)
    page.update()

def validate_integer(e):
    try:
        int(e.control.value)
    except ValueError:
        e.control.value = ""

def validate_grade(e):
    try:
        value = int(e.control.value)
        if value not in [3, 4, 5]:
            e.control.value = ""
    except ValueError:
        e.control.value = ""

def validate_oge_ege(e):
    try:
        value = int(e.control.value)
        if value < 0 or value > 100:
            e.control.value = ""
    except ValueError:
        e.control.value = ""

def update_average(fields_left, fields_right, subjects_to_average, page):
    total = 0
    count = 0
    for subject in subjects_to_average:
        if subject in fields_left:
            value = fields_left[subject].value
        else:
            value = fields_right[subject].value
        
        if value:
            try:
                total += float(value)
                count += 1
            except ValueError:
                pass
    
    if count > 0:
        average = total / count
    else:
        average = 0
    
    fields_right["Certificate"].value = f"{average:.2f}"
    page.update()

def collect_data(fields_left, fields_right):
    data = []
    features = [
        'Certificate', 'Algebra', 'Geometry', 'Physics', 'Computer Science', 'Russian Language',
        'Literature', 'History', 'Biology', 'Chemistry', 'Social Studies', 'Geography',
        'Physical Education', 'Foreign Language', 'Safety Basics', 'Music', 'Art', 'Technology',
        'Physics(EGE)', 'Math(EGE)', 'Math(OGE)'
    ]
    for feature in features:
        if feature in fields_left:
            data.append(float(fields_left[feature].value))
        else:
            data.append(float(fields_right[feature].value))
    return data, features

def show_results(page, fields_left, fields_right, subjects_to_average):
    data, features = collect_data(fields_left, fields_right)
    data_df = pd.DataFrame([data], columns=features)
    
    scaler_path = r'data_generating\code\predict_model\SMOTE_data\scaler.pkl'
    scaler = joblib.load(scaler_path)
    
    normalized_data = scaler.transform(data_df)
    normalized_data_df = pd.DataFrame(normalized_data, columns=features)
    
    model_prepod_path = r'data_generating\code\predict_model\SMOTE_data\random_forest\model_prepod_rf.pkl'
    model_vkr_path = r'data_generating\code\predict_model\SMOTE_data\random_forest\model_vkr_rf.pkl'

    model_prepod = joblib.load(model_prepod_path)
    model_vkr = joblib.load(model_vkr_path)
    
    prediction_prepod = model_prepod.predict(normalized_data_df)[0]
    prediction_vkr = model_vkr.predict(normalized_data_df)[0]
    
    accuracy_prepod = 0.969620253164557
    accuracy_vkr = 0.90715935334873

    recommendations = generate_recommendations(prediction_prepod, prediction_vkr)

    results_page(page, prediction_prepod, prediction_vkr, accuracy_prepod, accuracy_vkr, recommendations)

def generate_recommendations(prediction_prepod, prediction_vkr):
    recommendations = ""
    if prediction_prepod >= 90 and prediction_vkr >= 90:
        recommendations = "Отличные результаты! Продолжайте в том же духе!"
    elif prediction_prepod >= 75 and prediction_vkr >= 75:
        recommendations = "Хорошие результаты, но есть над чем поработать."
    else:
        recommendations = "Есть проблемы с успеваемостью. Рекомендуем обратиться за дополнительной помощью."
    return recommendations

ft.app(target=main)
