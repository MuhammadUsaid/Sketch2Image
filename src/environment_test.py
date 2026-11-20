
import sys
import numpy as np
import cv2
import PIL
import matplotlib
import pandas as pd

print("Sketch2Image Environment Test")
print("-" * 40)

print("Python:", sys.version)
print("Executable:", sys.executable)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pd.__version__)

print("-" * 40)
print("Environment configured successfully!")
