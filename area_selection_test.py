# import aspose.words as aw
# import streamlit as st
# # Create and save a simple document
# doc = aw.Document("D:\Projects\LPMS\docs\Annual Parks Bid.pdf")
# doc.save("OutputABC.md")
# with open("OutputABC.md", "r", encoding="utf-8") as markdown_file:
#         markdown_content = markdown_file.read()

#         # Display the content using the st.markdown() function
#         st.markdown(markdown_content)

# import streamlit as st
# import pdfplumber

# with pdfplumber.open("D:\Projects\LPMS\docs\Annual Parks Bid.pdf") as f:
#     for i in f.pages:
#         if(i.extract_tables()):
#             st.write((i.extract_table())[0])

# import tkinter as tk

# class AreaSelectorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Area Selector")
        
#         self.canvas = tk.Canvas(self.root, bg="white")
#         self.canvas.pack(fill=tk.BOTH, expand=True)
        
#         self.start_x, self.start_y = None, None
#         self.end_x, self.end_y = None, None
        
#         self.canvas.bind("<ButtonPress-1>", self.on_button_press)
#         self.canvas.bind("<B1-Motion>", self.on_button_drag)
#         self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
#     def on_button_press(self, event):
#         self.start_x, self.start_y = event.x, event.y
        
#     def on_button_drag(self, event):
#         self.end_x, self.end_y = event.x, event.y
#         self.draw_rectangle()
        
#     def on_button_release(self, event):
#         self.end_x, self.end_y = event.x, event.y
#         self.draw_rectangle()
        
#     def draw_rectangle(self):
#         self.canvas.delete("rectangle")
#         self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="rectangle")


# root = tk.Tk()
# app = AreaSelectorApp(root)
# root.mainloop()


import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
Image.MAX_IMAGE_PIXELS = None


# Specify canvas parameters in application
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
# )

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# if drawing_mode == 'point':
#     point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

bg_image_pil = Image.open(bg_image) if bg_image else None

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=1,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=bg_image_pil,
    update_streamlit=realtime_update,
    height=bg_image_pil.height if bg_image_pil else 500,
    width=bg_image_pil.width if bg_image_pil else 600,
    drawing_mode="rect",
    key="canvas",
    point_display_radius=0,  # Adjust as needed
    display_toolbar=True,   # Display the toolbar
)

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)
if canvas_result.image_data is not None and canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    print(objects)
    if not objects.empty and objects.iloc[0]["type"] == "rect":
        selected_rectangle = objects.iloc[0]

        if bg_image:
            image = Image.open(bg_image)
            image_width, image_height = image.size

            left = int(selected_rectangle["left"] * image_width)
            top = int(selected_rectangle["top"] * image_height)
            width = int(selected_rectangle["width"] * image_width)
            height = int(selected_rectangle["height"] * image_height)

            # Print coordinates and dimensions for debugging
            print(f"Left: {left}, Top: {top}, Width: {width}, Height: {height}")

            # Crop the image using the selected rectangle
            cropped_image = image.crop((left, top, left + width, top + height))
            st.image(cropped_image, caption="Cropped Image", use_column_width=True)
            
            # Save the cropped image
            save_path = "cropped_image.jpg"
            cropped_image.save(save_path)
            st.success(f"Cropped image saved as {save_path}")