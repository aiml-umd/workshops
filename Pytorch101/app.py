import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io

# Define the canvas size (28x28 grid for MNIST)
canvas_size = 28

if "model" not in st.session_state:
    st.session_state.model = None

# Create a 28x28 grid where user can draw
def create_canvas():
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    return canvas

# Display the grid in the Streamlit app
def display_grid(canvas):
    st.write("Draw on the grid:")

    # Create a 28x28 grid using Streamlit columns
    for i in range(canvas_size):
        cols = st.columns(canvas_size)
        for j, col in enumerate(cols):
            # Each button toggles the pixel state between 0 (off) and 1 (on)
            if col.button(' ', key=f'{i}-{j}', help=f'Cell {i},{j}', use_container_width=True):
                canvas[i, j] = 1 if canvas[i, j] == 0 else 0  # Toggle between 0 and 1
    return canvas

# Predict the digit from the grid using the uploaded PyTorch JIT model
def predict_digit(canvas):
    # Normalize and reshape the canvas to fit the model's expected input (1x1x28x28)
    input_tensor = torch.tensor(canvas).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    input_tensor = input_tensor.to(torch.float32)  # Ensure it's in float32 for the model

    model = st.session_state.model
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()

    return predicted_digit

# Main Streamlit app function
def main():
    st.title("Draw a Digit on the 28x28 Grid (Upload Your PyTorch Model)")
    
    # Check if a file has been uploaded
    if st.session_state.model:

        # Step 2: Create a canvas where the user draws
        canvas = create_canvas()

        # Step 3: Display the grid and let the user draw
        canvas = display_grid(canvas)

        # Convert the canvas into an image for better visualization
        canvas_img = Image.fromarray(np.uint8(canvas * 255), 'L')
        st.image(canvas_img.resize((280, 280)), caption="Your Drawing", use_column_width=True)

        # Step 4: Predict the digit based on the drawn grid
        if st.button("Predict"):
            prediction = predict_digit(canvas)
            st.success(f"Predicted Digit: {prediction}")
    else:
        uploaded_file = st.file_uploader("Upload your JIT-compiled PyTorch model", type=["pt"])
        if uploaded_file:
            st.session_state.model = torch.jit.load(uploaded_file)
            st.rerun()

if __name__ == "__main__":
    main()
