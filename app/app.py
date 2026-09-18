import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)


# ==========================================
# Classes
# ==========================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    model = mobilenet_v3_small(weights=None)

    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        4
    )

    checkpoint = torch.load(
        "models/tomato_disease_mobilenetv3.pth",
        map_location="cpu"
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


model = load_model()


# ==========================================
# Image Transformation
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Title
# ==========================================

st.title("🍅 Tomato Disease Detection")

st.write(
    "Upload a tomato leaf image and the AI model "
    "will predict the possible disease."
)


# ==========================================
# Upload Image
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        # Prepare image
        input_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )

        predicted_label = CLASS_NAMES[
            predicted_class.item()
        ]

        confidence_value = confidence.item() * 100


        # ==========================================
        # Display Result
        # ==========================================

        st.subheader("Prediction")

        if predicted_label == "Tomato___healthy":

            st.success("🌿 Tomato Leaf: Healthy")

        else:

            disease_name = predicted_label.replace(
                "Tomato___",
                ""
            ).replace(
                "_",
                " "
            )

            st.error(
                f"🦠 Disease Detected: {disease_name}"
            )

        st.info(
            f"Confidence: {confidence_value:.2f}%"
        )


        # ==========================================
        # Show All Probabilities
        # ==========================================

        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(CLASS_NAMES):

            probability = (
                probabilities[0][i].item() * 100
            )

            display_name = class_name.replace(
                "Tomato___",
                ""
            ).replace(
                "_",
                " "
            )

            st.write(
                f"{display_name}: "
                f"{probability:.2f}%"
            )

            st.progress(
                probability / 100
            )