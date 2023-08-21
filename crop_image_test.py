import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import pytesseract
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Preview")
    st.image(cropped_img, use_column_width=True)
    if st.button("Crop Image"):
        save_path = "cropped_image.jpg"
        cropped_img_rgb = cropped_img.convert("RGB")
        cropped_img_rgb.save(save_path, format="JPEG")
        st.success(f"Cropped image saved as {save_path}")
        # extracted_text = pytesseract.image_to_string(cropped_img_rgb)
        # st.subheader("Extracted Text:")
        # st.write(extracted_text)