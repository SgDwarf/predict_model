import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Загрузка данных
file_path = r'data_generating\files\SMOTE generated\final_generated_students.csv'  # Укажите путь к вашему CSV файлу с данными
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

# Проверка наличия всех столбцов
missing_columns = [col for col in features if col not in data.columns]
if missing_columns:
    print(f'Missing columns: {missing_columns}')
else:
    # Нормализация данных
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    # Сохранение нормализованных данных в новый CSV файл
    normalized_file_path = r'data_generating\code\predict_model\SMOTE_data\normalized_data.csv'
    data.to_csv(normalized_file_path, index=False, encoding='utf-8')

    # Сохранение скалера
    scaler_file_path = r'data_generating\code\predict_model\SMOTE_data\scaler.pkl'
    joblib.dump(scaler, scaler_file_path)

    print(f'Data successfully saved to {normalized_file_path}')
    print(f'Scaler successfully saved to {scaler_file_path}')
