# src/inference.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# === Настройки ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Классы болезней ===
TOMATO_CLASSES = [
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Yellow Leaf Curl Virus',
    'Healthy'
]

CUCUMBER_CLASSES = [
    'Bacterial Wilt',
    'Downy Mildew',
    'Gummy Stem Blight',
    'Healthy',
    'Powdery Mildew',
    'Angular Leaf Spot'
]

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# === Пути к моделям ===
SPECIES_MODEL_PATH = "models/convnext_tiny_species_classifier.pth"
TOMATO_MODEL_PATH = "models/vit_base_tomato_disease.pth"
CUCUMBER_MODEL_PATH = "models/vit_base_cucumber_disease.pth"


def load_species_model():
    """Загружает модель для определения вида: tomato vs cucumber"""
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(768, 2)  # 2 класса
    model.load_state_dict(torch.load(SPECIES_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_vit_model(num_classes, model_path):
    """Загружает ViT-модель с правильной структурой head"""
    # Загружаем базовую архитектуру
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)

    # Заменяем head на Linear(768, num_classes)
    model.head = nn.Linear(768, num_classes)

    # Загружаем веса
    state_dict = torch.load(model_path, map_location=DEVICE)

    # ВАЖНО: если в сохранённой модели head называется head.1 — исправим
    new_state_dict = {}
    for k, v in state_dict.items():
        # Если в сохранённой модели head — это Sequential с одним линейным слоем
        if k.startswith("head.1."):
            new_key = "head." + k.split("head.1.")[-1]  # head.1.weight → head.weight
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_species(model, image_tensor):
    """Определяет, томат или огурец"""
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    species = "Tomato" if pred == 0 else "Cucumber"
    return species, conf


def predict_disease(model, image_tensor, class_names):
    """Предсказывает болезнь"""
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    disease = class_names[pred]
    return disease, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    args = parser.parse_args()

    image_path = args.image

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    # --- Загрузка изображения ---
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # --- Этап 1: Определение вида ---
    print("Определяем: томат или огурец...")
    species_model = load_species_model()
    species, species_conf = predict_species(species_model, image_tensor)
    print(f"Вид: {species} (уверенность: {species_conf:.3f})")

    # --- Этап 2: Загрузка нужной ViT-модели ---
    print(f"Загружаем модель для {species}...")
    if species == "tomato":
        disease_model = load_vit_model(len(TOMATO_CLASSES), TOMATO_MODEL_PATH)
        class_names = TOMATO_CLASSES
    else:
        disease_model = load_vit_model(len(CUCUMBER_CLASSES), CUCUMBER_MODEL_PATH)
        class_names = CUCUMBER_CLASSES

    # --- Этап 3: Предсказание болезни ---
    disease, disease_conf = predict_disease(disease_model, image_tensor, class_names)
    print(f"Диагноз: {disease} (уверенность: {disease_conf:.3f})")

    # --- Визуализация ---
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Вид: {species}\nБолезнь: {disease}\nУверенность: {disease_conf:.3f}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()