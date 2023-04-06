import streamlit as st
import cv2
import numpy as np
from PIL import Image

# st.set_page_config(layout="wide")

st.title("Image Anonymizer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # OpenCV reads images in BGR format
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to fit the canvas size
    img = cv2.resize(img, (400, 400))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Select region to anonymize"):
        # Create a canvas with the uploaded image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas = np.copy(img)

        # Create a rectangle selector using OpenCV
        r = cv2.selectROI(canvas)

        # Show the selected region with a red rectangle
        cv2.rectangle(canvas, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), (255, 0, 0), 2)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = np.copy(canvas)
        st.image(canvas, caption="Region selected", use_column_width=True)

        # Apply Laplacian noise to the selected region
        epsilon = 0.005
        mask = np.zeros_like(img)
        mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 1
        noise = np.random.laplace(0, 1/epsilon, (400, 400, 3))
        anonymized = np.where(mask == 1, np.clip(img + noise, 0, 255).astype(np.uint8), img)

        anonymized = cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)
        anonymized = np.copy(anonymized)
        st.image(anonymized, caption="Anonymized image", use_column_width=True)

        # Provide option to download output image
        # im_pil = Image.fromarray(anonymized)
        # im_bytes = im_pil.tobytes()
        # st.download_button(label="Download Anonymized Image", data=im_bytes, file_name="anonymized_image.jpeg")
