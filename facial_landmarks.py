"""
Facial Landmarks Pro — MediaPipe Tasks API (478 points)
Author: Yuseph Alvandi
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from PIL import Image, ImageTk
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FacialLandmarksApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Facial Landmarks Pro")
        self.window.geometry("1100x800")
        self.window.configure(fg_color="#0a0a0a")
        
        self.image_path = None
        self.original_img = None
        self.current_img = None
        
        base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.3
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
        self.setup_ui()
    
    def setup_ui(self):
        header = ctk.CTkFrame(self.window, fg_color="transparent")
        header.pack(fill="x", pady=(20, 10), padx=30)
        ctk.CTkLabel(header, text="Facial Landmarks Pro", font=ctk.CTkFont(size=32, weight="bold"), text_color="#1E90FF").pack()
        ctk.CTkLabel(header, text="478-point facial mesh using MediaPipe", font=ctk.CTkFont(size=14), text_color="#AAAAAA").pack()
        
        toolbar = ctk.CTkFrame(self.window, fg_color="#1a1a1a", corner_radius=12, height=60)
        toolbar.pack(fill="x", padx=30, pady=(5, 10))
        toolbar.pack_propagate(False)
        ctk.CTkButton(toolbar, text="Open Image", command=self.open_image, width=150, height=40).pack(side="left", padx=(20, 10), pady=10)
        ctk.CTkButton(toolbar, text="Save Result", command=self.save_result, width=150, height=40, fg_color="#2ECC71").pack(side="left", padx=10, pady=10)
        self.file_label = ctk.CTkLabel(toolbar, text="No image selected", text_color="#888888")
        self.file_label.pack(side="left", padx=20, pady=10)
        
        content = ctk.CTkFrame(self.window, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=10)
        right = ctk.CTkFrame(content, fg_color="#1a1a1a", corner_radius=12)
        right.pack(fill="both", expand=True)
        self.preview_label = ctk.CTkLabel(right, text="No Image Loaded", font=ctk.CTkFont(size=16), text_color="#555555")
        self.preview_label.pack(expand=True)
    
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.image_path = path
        self.original_img = cv2.imread(path)
        self.file_label.configure(text=os.path.basename(path))
        self.detect_landmarks()
    
    def detect_landmarks(self):
        if self.original_img is None: return
        img = self.original_img.copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)
        
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                for lm in face_landmarks:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
        
        self.current_img = img
        rgb_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_disp)
        max_w, max_h = 700, 600
        ratio = min(max_w / pil_img.width, max_h / pil_img.height)
        pw, ph = int(pil_img.width * ratio), int(pil_img.height * ratio)
        pil_img = pil_img.resize((pw, ph))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img
    
    def save_result(self):
        if self.current_img is None: messagebox.showerror("Error", "No processed image!"); return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path: cv2.imwrite(path, self.current_img)
    
    def run(self): self.window.mainloop()

if __name__ == "__main__":
    app = FacialLandmarksApp()
    app.run()