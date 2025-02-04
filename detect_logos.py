import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from ultralytics import YOLO
from utils import get_transforms
import yaml

n = 120  # заменить на действительное число классов

model_classifier = models.resnet50(pretrained=False)

model_classifier.fc = nn.Linear(model_classifier.fc.in_features, n)
model_classifier.load_state_dict(torch.load('models/logo_classifier.pth'))
model_classifier.eval()  # Переводим модель в режим инференса
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Проверяем доступность GPU
model_classifier.to(device)
classifier_transform = get_transforms()

# Загрузка обученного детектора YOLOv8
model_detector = YOLO('models/yolo8_detector.pt')

with open('data/yolo_data.yaml', 'r') as f:
    data = yaml.safe_load(f)


def detect_and_classify(image_path, name):
    """
    Обнаруживает и классифицирует логотипы на изображении.

    Args:
        image_path (str): Путь к изображению.
    """

    image = cv2.imread(image_path)
    results = model_detector(image)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_logo = image[y1:y2, x1:x2]

            # Классифицируем вырезанный логотип
            cropped_logo_tensor = classifier_transform(cropped_logo).unsqueeze(0).to(device)
            with torch.no_grad():
                classification_result = torch.argmax(model_classifier(cropped_logo_tensor))

            # Получаем название класса (компании) по индексу
            class_name = data['names'][classification_result]

            if class_name == name:
                print(f"Обнаружен и классифицирован логотип: {class_name} at: {x1, y1, x2, y2}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

    cv2.imshow("Detected Logos", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Пример использования:
image_path = 'path/to/your/image.jpg'  # Замените на путь к вашему изображению
name = ''  # сюда ввести название компании
detect_and_classify(image_path, name)
