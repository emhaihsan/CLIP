# models/clip_model.py
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_match(image, text_input):
    # Pisahkan deskripsi berdasarkan koma
    descriptions = [desc.strip() for desc in text_input.split(',')]
    
    # Preprocess gambar dan teks
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True).to(device)
    
    # Hitung kemiripan antara gambar dan deskripsi
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Skor kesesuaian gambar ke teks
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]  # Konversi ke probabilitas
    
    # Membuat string untuk menampilkan deskripsi beserta probabilitasnya
    result = ""
    for desc, prob in zip(descriptions, probs):
        result += f"Description: '{desc}' - Probability: {prob:.4f}\n"
    
    return result
