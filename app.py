import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

@st.cache_resource  # чтобы не перезагружать модель при каждом перезапуске
def load_model(model_path="best_model.pth"):
    # Пример, если это ResNet18 c 2 выходами
    import torchvision.models as models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model("best_model.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Добавьте Normalize, если вы тренировались на ImageNet-стандарте
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

st.title("Классификация рентген-снимков")

uploaded_file = st.file_uploader("Загрузите снимок", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Загруженный снимок", use_column_width=True)

    # Предобработаем изображение (resize, to tensor, normalize)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = model(input_batch)
        _, predicted = torch.max(outputs, 1)

    classes = ['NORMAL', 'PNEUMONIA']
    result = classes[predicted.item()]

    st.write(f"**Результат классификации**: {result}")



st.sidebar.title("Меню")
st.sidebar.write("Тут можно добавлять разные настройки, фильтры")
