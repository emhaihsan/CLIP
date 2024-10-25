# CLIP Image-Text Matching with Gradio

This project is an end-to-end application that leverages OpenAI's CLIP model to match images with multiple text descriptions and return the matching probabilities. The interface is built using Gradio, allowing users to upload an image and input multiple text descriptions. The output is a list of descriptions with their corresponding matching probabilities.

## Features

- Upload an image and input multiple text descriptions.
- The CLIP model computes the similarity between the image and each text description.
- Returns a list of text descriptions with their corresponding matching probabilities.
- Simple interface built with Gradio for easy interaction.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   You will need to install the required libraries listed in the `requirements.txt` file:

   ```bash
   pip install torch torchvision transformers gradio
   ```

4. **Install Matplotlib** (if needed for future visualization):

   ```bash
   pip install matplotlib
   ```

## Usage

1. **Run the application:**

   ```bash
   python app.py
   ```

2. **How to use:**

   - Upload an image in the interface.
   - Enter multiple text descriptions separated by commas (e.g., "a dog, a cat, a car").
   - The application will return a list of descriptions along with their matching probabilities to the uploaded image.

## File Structure

```plaintext
.
├── app.py                  # Main file to run the Gradio app
├── clip_model.py           # Contains the CLIP matching logic
├── requirements.txt        # List of required dependencies
└── README.md               # Project documentation
```

## Code Overview

- `app.py`: This file sets up the Gradio interface. Users can upload an image and input text descriptions, and the matching probabilities will be displayed as text output.
- `clip_model.py`: This file contains the function `clip_match`, which processes the image and text input, uses the CLIP model to compute similarity scores, and returns the probabilities in a human-readable format.

## Example

1. **Upload an image**

2. **Enter text descriptions:**

   ```plaintext
   "a cat, a dog, a person"
   ```

3. **Output:**

   ```plaintext
   Description: 'a cat' - Probability: 0.8234
   Description: 'a dog' - Probability: 0.1209
   Description: 'a person' - Probability: 0.0557
   ```
