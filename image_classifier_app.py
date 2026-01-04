import streamlit as st


# Step 1: Configure Streamlit page
st.set_page_config(page_title="CPU Image Classifier", layout="centered")
st.title("CPU-Based Image Classification using ResNet-18")

# Step 2 & 3: Import libraries & CPU
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
device = torch.device("cpu")


# Step 4: Load Pretrained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
model.to(device)


# Step 5: Load recommended image transformations
transform = weights.transforms()


# Step 6: File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load ImageNet labels
labels = weights.meta["categories"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    # Step 7: Convert image to tensor; .Run inference (no gradients)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

    
    # Step 8: Apply Softmax
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append([labels[top5_idx[i]], float(top5_prob[i] * 100)])

    df = pd.DataFrame(results, columns=["Class", "Probability (%)"])

    st.subheader("Top-5 Predictions")
    st.dataframe(df)

    # Step 9: Bar chart
    st.subheader("Prediction Probability Distribution")
    st.bar_chart(df.set_index("Class"))
