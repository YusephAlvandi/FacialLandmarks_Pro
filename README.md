Facial Landmarks Pro

A deep learning-based application for detecting 478-point facial mesh in images. Uses MediaPipe's Face Landmarker model—a deep neural network trained on thousands of annotated face images. Unlike older apps using classical models like Haar Cascade (bounding boxes only) or LBF (68 points), this provides 7x more detail.


WHAT IT DOES

This application identifies human faces in images and maps 478 precise landmarks onto each face; including eyes, eyebrows, nose, lips, and facial contour. In the output green dots are shown on every detected landmark point. The output can be saved as a new image.
It runs completely offline on your local storage, with no privacy concern.


COMPETITIVE ADVANTAGE

Compared to older facial landmark tools:

- 478 points vs. 68 points (LBF) (7x more detail)
- Deep learning vs. classical ML (higher accuracy)
- Detects faces at multiple angles, not just frontal
- Works offline (your images never leave your computer)
- Built by a physicist with expertise in optics and image processing


DEPENDENCIES

Python libraries (install once):
pip install mediapipe opencv-python pillow customtkinter

Model file (one-time download):
face_landmarker.task (3.6 MB) — place in the same folder as the script


HOW TO RUN

python facial_landmarks.py

1. Click Open Image and select the desired image to be processed.
2. The 478-point facial mesh is automatically detected and shown.
3. Click Save Result to save the labeled image.


AUTHOR

Yuseph Alvandi
PhD in Optics and Laser Physics
Python Developer and Image Processing Specialist

GitHub: https://github.com/YusephAlvandi


LICENSE

MIT License
