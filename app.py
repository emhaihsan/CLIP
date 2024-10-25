# app.py
import gradio as gr
from clip_model import clip_match

def main():
    image_input = gr.Image(type="pil", label="Upload an image")
    text_input = gr.Textbox(lines=2, placeholder="Enter text descriptions, separated by commas", label="Enter Descriptions")
    
    # Gunakan output teks untuk menampilkan probabilitas
    output = gr.Textbox(label="Matching Probabilities")

    gr.Interface(fn=clip_match, 
                 inputs=[image_input, text_input], 
                 outputs=output, 
                 title="CLIP Image-Text Matching",
                 description="Upload an image and enter multiple text descriptions separated by commas. CLIP will return the matching probabilities."
                ).launch()

if __name__ == "__main__":
    main()
