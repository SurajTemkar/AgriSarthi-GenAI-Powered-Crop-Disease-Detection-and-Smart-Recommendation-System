import streamlit as st
st.set_page_config(page_title="Farmer’s Decision Support System", layout="wide")
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from vit_model import VisionTransformer
import pandas as pd
from PIL import Image, ImageEnhance
import google.generativeai as genai
from deep_translator import GoogleTranslator
 # Ensure this function is properly defined
from googletrans import Translator 


# Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyCoERT0gIeu8zYWzUT1Z0nNNlnBwvkx8Bw"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

def translate_text(text, target_language="en"):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text # If translation fails, return the original text

# Path to the uploaded logo
logo_path = "C:/Users/91932/Desktop/SYDS/RD/jaymataji/Screenshot_26-2-2025_164927_www.agrisarathi.com-removebg-preview.png"

# Center the logo using Streamlit's built-in layout features
col1, col2, col3 = st.columns([1, 3, 1])  # Create three columns for centering
with col2:  # Put the image in the middle column
    st.image(logo_path, width=300)

# Load class names
CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ✅ Cache model loading (only loads once)
@st.cache_resource
def load_model():
    model = VisionTransformer(img_size=128, patch_size=8, num_classes=len(CLASS_NAMES),
                              embed_dim=768, depth=8, num_heads=12, mlp_dim=2048, dropout=0.1)
    model.load_state_dict(torch.load("C:/Users/91932/Desktop/SYDS/RD/jaymataji/models/custom_vit.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model() 

# ✅ Cache AI inference
def vit_inference(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    return CLASS_NAMES[predicted_idx], confidence
    # ✅ Generate AI response using Gemini and translate dynamically
def generate_gemini_response(disease_name, target_language):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"""
        You are an expert plant pathologist. The detected crop disease is: {disease_name}.
        Provide details on:
        - Pathogen details
        - Severity level
        - Symptoms
        - Economic impact
        - Treatment options (short-term and long-term)
        - Prevention strategies
        """
        response = model.generate_content(prompt)
        translated_response = translate_text(response.text, target_language)
        return translated_response
    except Exception:
        return translate_text("Error connecting to Gemini API.", target_language)

# 🔹 Function for Text Translation
def translate_text(text, target_language="mr"):
    try:
        return GoogleTranslator(source='auto', target=target_language).translate(text)
    except Exception:
        return text

# Sidebar for language selection
st.sidebar.title("🌍 Select Language")
language_options = {"English": "en", "Marathi": "mr"}
selected_language = st.sidebar.selectbox("Choose your language:", list(language_options.keys()), key="language")
selected_language_code = language_options[selected_language]

# UI Translations
translations = {
    "en": {
        "title": "Farmer's Decision Support System",
        "description": "Welcome to the AI-powered farming assistant!",
        "upload_image": "📤 Upload a plant image",
        "detected_disease": "🏷️ AI Detected Disease",
        "confidence_score": "📊 Confidence Score",
        "recommendation": "🌿 AI Expert Recommendation:",
        "feedback": "Feedback",
        "submit_feedback": "Submit Feedback"
    },
    "mr": {
        "title": "शेतकऱ्यांचा निर्णय समर्थन प्रणाली",
        "description": "AI-शक्तिशाली शेती सहाय्यकामध्ये आपले स्वागत आहे!",
        "upload_image": "📤 वनस्पतीची प्रतिमा अपलोड करा",
        "detected_disease": "🏷️ AI ओळखलेला रोग",
        "confidence_score": "📊 विश्वास गुण",
        "recommendation": "🌿 AI तज्ञाची शिफारस:",
        "feedback": "अभिप्राय",
        "submit_feedback": "अभिप्राय सबमिट करा"
    }
}
selected_translations = translations[selected_language_code]

# Dummy translation function (Replace this with your actual translation function)
def translate_text(text, target_language):
    # This is a placeholder function. Replace it with your actual translation logic.
    return text if target_language == "en" else f"[{text} in {target_language}]"
# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Sidebar Navigation ---
st.sidebar.title("🌿 Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
if st.sidebar.button("🎮 AI Challenge"):
    st.session_state.page = "AI Challenge"
if st.sidebar.button("📊 Training Progress"):
    st.session_state.page = "Training"
if st.sidebar.button("🎥 Video Demo"):
    st.session_state.page = "Video"
if st.sidebar.button("💬 Feedback"):
    st.session_state.page = "Feedback"
# UI Layout
st.title(selected_translations["title"])
st.write(selected_translations["description"])

# Initialize session state variables if not already set
if "predicted_class" not in st.session_state:
    st.session_state["predicted_class"] = None
if "confidence" not in st.session_state:
    st.session_state["confidence"] = None

# Define UI elements based on selected language
selected_language = st.session_state.get("language", "en")  # Default language is English

if st.session_state.page == "Home":
    st.title(translate_text("🌱AI-Powered Crop Disease Detection For Farmers👨‍🌾", selected_language))
    video_url = "https://drive.google.com/file/d/1MuTnF2IRMziqtv4SvHWekuoRmPUI6TiY/view?usp=sharing"
    st.markdown(f'🎥 [{translate_text("Want to Hear the Real Stories of Farmers? Click to watch!", selected_language)}]({video_url})', unsafe_allow_html=True)
    st.write(translate_text("#### AI helps farmers detect crop diseases early and increase yield!", selected_language))

elif st.session_state.page == "AI Challenge":
    st.title(translate_text("🎮 Want to See Our Model in Action?", selected_language))
    uploaded_file = st.file_uploader(translate_text("📤 Upload a plant image", selected_language), type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=translate_text("Uploaded Image", selected_language), width=300)
        
        predicted_class, confidence = vit_inference(image)
        st.write(f"### 🏷️ {translate_text('AI Detected Disease', selected_language)}: {predicted_class}")
        st.write(f"### 📊 {translate_text('Confidence Score', selected_language)}: {confidence * 100:.2f}%")
        
        recommendation = generate_gemini_response(predicted_class, selected_language)
        st.subheader(translate_text("🌿 AI Expert Recommendation:", selected_language))
        st.write(recommendation)
        
        st.session_state["predicted_class"] = predicted_class
        st.session_state["confidence"] = confidence
        
        data = {
            "Feature": [translate_text("🏷️ AI Detected Disease", selected_language), translate_text("📊 Confidence Score", selected_language)],
            "Result": [predicted_class, f"{confidence * 100:.2f}%"]
        }
        
        df = pd.DataFrame(data)
        st.table(df)

elif st.session_state.page == "Training":
    st.title(translate_text("📊 Model Accuracy & Training Progress", selected_language))
    
    # Display model comparison image
    st.image("C:/Users/91932/Desktop/SYDS/RD/jaymataji/Screenshot_26-2-2025_162131_.jpeg", 
             caption=translate_text("ViT Model Comparison", selected_language), width=600)
    
    st.subheader(translate_text("🧠 ViT Model - Training Performance", selected_language))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("C:/Users/91932/Desktop/SYDS/RD/jaymataji/acc.jpeg", 
                 caption=translate_text("Training Accuracy", selected_language), width=300)
    with col2:
        st.image("C:/Users/91932/Desktop/SYDS/RD/jaymataji/loss.jpeg", 
                 caption=translate_text("Loss Reduction", selected_language), width=300)

elif st.session_state.page == "Video":
    st.title(translate_text("🌱 AI Detects Crop Diseases in Seconds! Watch Now!! 🎥", selected_language))
    st.video("C:/Users/91932/Desktop/SYDS/RD/jaymataji/FINAL.mp4")
    st.write(translate_text("🎥 See how AI identifies crop diseases in real time!", selected_language))

# ✅ Feedback Section in Sidebar
st.sidebar.title(translate_text("Feedback", selected_language))
feedback = st.sidebar.text_area(translate_text("Enter your feedback here...", selected_language))

if st.sidebar.button(translate_text("Submit Feedback", selected_language)):
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"{feedback}\n")
    st.sidebar.success(translate_text("✅ Feedback submitted!", selected_language))

# 🔑 Admin Password Field (ONLY in Feedback Section)
with st.expander("🔑 " + translate_text("Admin Access", selected_language)):
    admin_access = st.text_input(translate_text("Enter Admin Password", selected_language), type="password", key="admin_password")

    ADMIN_PASSWORD = "admin123"
    
    if admin_access == ADMIN_PASSWORD:
        st.success(translate_text("🔓 Access Granted! Here is the feedback:", selected_language))
        try:
            with open("feedback.txt", "r", encoding="utf-8") as f:
                feedback_history = f.readlines()
            if feedback_history:
                for idx, fb in enumerate(feedback_history, 1):
                    st.write(f"**{idx}.** {fb.strip()}")
            else:
                st.info(translate_text("No feedback submitted yet.", selected_language))
        except FileNotFoundError:
            st.info(translate_text("No feedback submitted yet.", selected_language))
    elif admin_access:
        st.error(translate_text("❌ Incorrect Password! Access Denied.", selected_language))