import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import os

# Device setup: use CPU if no GPU available
device = torch.device("cpu")
print(f"Using device: {device}")

# Define the GenderModel class
class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(5)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(5)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc6 = nn.Linear(13824, 512)

        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(512, 512)

        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.norm1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.norm2(x)
        x = self.pool5(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.relu6(self.fc6(x))
        x = self.drop6(x)
        x = self.relu7(self.fc7(x))
        x = self.drop7(x)
        x = self.fc8(x)

        return self.softmax(x)

# Instantiate and load the GenderModel
GenderModel = EnhancedNet().to(device)
if os.path.exists("GenderModel.pth"):
    GenderModel.load_state_dict(torch.load("GenderModel.pth", map_location=device))
else:
    print("GenderModel.pth not found!")

# Define the AgeModel class
class EnhancedNet2(nn.Module):
    def __init__(self):
        super(EnhancedNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(5)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(5)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc6 = nn.Linear(13824, 512)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(512, 512)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.5)

        self.fc8 = nn.Linear(512, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.norm1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.norm2(x)
        x = self.pool5(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.relu6(self.fc6(x))
        x = self.drop6(x)
        x = self.relu7(self.fc7(x))
        x = self.drop7(x)
        x = self.fc8(x)
        return self.softmax(x)

# Instantiate and load the AgeModel
AgeModel = EnhancedNet2().to(device)
if os.path.exists("AgeModel.pth"):
    AgeModel.load_state_dict(torch.load("AgeModel.pth", map_location=device))
else:
    print("AgeModel.pth not found!")

# Define preprocessing transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess image
def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Gender and Age Prediction")
st.write("Upload an image to predict the gender and age.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    # Process the uploaded image
    image = process_image(uploaded_file)

    # Gender prediction
    GenderModel.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        gender_output = GenderModel(image.to(device))
        gender_pred = torch.argmax(gender_output, dim=1).item()
        gender = "Male" if gender_pred == 1 else "Female"
    
    # Age prediction
    AgeModel.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        age_output = AgeModel(image.to(device))
        age_pred = torch.argmax(age_output, dim=1).item()
        age_group = "Young" if age_pred == 0 else "Middle" if age_pred == 1 else "Old"

    # Display results
    st.write(f"Predicted Gender: {gender}")
    st.write(f"Predicted Age Group: {age_group}")
