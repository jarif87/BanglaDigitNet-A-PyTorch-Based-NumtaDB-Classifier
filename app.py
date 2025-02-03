import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
n_classes = 10
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)

model.load_state_dict(torch.load("NumtaDB_Classifier_Model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_name = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Nine", "Ten"]

def predict(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    predictions = {label_name[i]: float(probs[0][i]) for i in range(len(label_name))}

    return predictions

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Label(num_top_classes=len(label_name)),
    title="BanglaDigitPro: Advanced Bengali Numeral Recognition",
    description="Upload an image of a handwritten Bangla digit to classify it.",
    examples=[["example_1.png"], ["example_2.png"], ["example_3.png"], ["example_4.png"], ["example_5.png"]])

iface.launch(share=True)
