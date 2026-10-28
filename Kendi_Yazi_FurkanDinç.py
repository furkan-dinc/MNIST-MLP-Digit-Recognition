import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        out = self.fc3(x)
        return out

target_dir = os.path.expanduser("~/Desktop/DerinOgrenmeOdevi")
home_dir = os.path.expanduser("~")

model_path = os.path.join(target_dir, 'mnist_mlp_model.pth')
if not os.path.exists(model_path):
    model_path = os.path.join(home_dir, 'mnist_mlp_model.pth')

model = MLPModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    img = img.resize((28, 28))
    img_np = np.array(img, dtype=np.float32)
    
    img_np = 255.0 - img_np
    
    p_low, p_high = np.percentile(img_np, (25, 98))
    if p_high > p_low:
        img_np = np.clip((img_np - p_low) / (p_high - p_low) * 255.0, 0, 255)
    
    img_np[img_np < 90] = 0
    
    img_clean = Image.fromarray(img_np.astype('uint8'))
    tensor_img = transform(img_clean)
    return img_clean, tensor_img

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_files = [
    f for f in os.listdir(target_dir)
    if f.lower().endswith(valid_extensions) and not f.startswith('kendi_el_yaziniz')
]
image_files.sort()

if not image_files:
    print("Resim dosyası bulunamadı!")
else:
    fig = plt.figure(figsize=(15, 6))
    cols = min(6, len(image_files))
    rows = (len(image_files) + cols - 1) // cols

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(target_dir, img_name)
        original_28x28, input_tensor = preprocess_image(img_path)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        print(f"Dosya: {img_name:<25} -> Model Tahmini: {prediction}")

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(original_28x28, cmap='gray')
        ax.set_title(f"Tahmin: {prediction}", fontsize=12, fontweight='bold', color='blue')
        ax.axis('off')

    plt.tight_layout()
    output_save_path = os.path.join(target_dir, 'kendi_el_yaziniz_sonuclari.png')
    plt.savefig(output_save_path, dpi=300)
    plt.show()