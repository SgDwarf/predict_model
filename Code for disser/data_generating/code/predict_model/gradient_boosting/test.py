import flet as ft

def main(page: ft.Page):
    page.title = "Визуализация тестирования моделей"
    page.window_width = 1200
    page.window_height = 900
    page.scroll = ft.ScrollMode.AUTO  # Включение прокрутки
    
    # Загрузка base64 строк из файлов
    with open("data_generating/code/predict_model/SMOTE_data/gradient_boosting/graph_image_prepod.txt", "r") as file:
        prepod_image_base64 = file.read()
    
    with open("data_generating/code/predict_model/SMOTE_data/gradient_boosting/graph_image_vkr.txt", "r") as file:
        vkr_image_base64 = file.read()
    
    # Flet компоненты для отображения метрик и графиков
    metrics_column = ft.Column([
        ft.Text(f"Prepod_mark - MSE: 0.45"),
        ft.Text(f"Prepod_mark - R2: 0.85"),
        ft.Text(f"Prepod_mark - MAE: 0.35"),
        ft.Text(f"VKR - MSE: 0.55"),
        ft.Text(f"VKR - R2: 0.75"),
        ft.Text(f"VKR - MAE: 0.45")
    ], alignment=ft.MainAxisAlignment.START, expand=True)
    
    graph_image_prepod = ft.Image(src_base64=prepod_image_base64, width=page.window_width, expand=True)
    graph_image_vkr = ft.Image(src_base64=vkr_image_base64, width=page.window_width, expand=True)
    
    page.add(
        ft.Column(
            [
                metrics_column,
                graph_image_prepod,
                graph_image_vkr
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True
        )
    )

ft.app(target=main)
