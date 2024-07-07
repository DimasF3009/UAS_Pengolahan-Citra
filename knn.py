import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def reshape_image(image):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    return pixel_values

def apply_kmeans(pixel_values, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_
    return labels, centers

def create_segmented_image(labels, centers, image_shape):
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_shape)
    return segmented_image

def calculate_color_percentage(labels, centers):
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)
    percentages = {tuple(np.uint8(centers[i])): count / total_pixels * 100 for i, count in zip(unique, counts)}
    return percentages

def display_color_percentages(percentages, col):
    for color, percentage in percentages.items():
        col.write(f"Color {color}: {percentage:.2f}%")
        col.write(
            f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: 50px; height: 50px;'></div>",
            unsafe_allow_html=True
        )

def main():
    st.title("Segmentasi Gambar dengan KNN")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)

        # Reshape and apply KMeans
        pixel_values = reshape_image(image)
        labels, centers = apply_kmeans(pixel_values, k)

        # Create segmented image
        segmented_image = create_segmented_image(labels, centers, image.shape)

        # Display original and segmented images side by side
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption='Gambar Asli', use_column_width=True)
        
        with col2:
            st.image(segmented_image, caption='Segmentasi Gambar', use_column_width=True)

        with col3:
            st.write("Persentasi segmentasi gambar:")
            percentages = calculate_color_percentage(labels, centers)
            display_color_percentages(percentages, col3)

if __name__ == "__main__":
    main()
