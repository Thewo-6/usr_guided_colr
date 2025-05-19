import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QColorDialog

# ---------- Global Drawing Color State ----------
selected_rgb = (255, 0, 0)
selected_ab = np.array([80, 70], dtype=np.float32)

# ---------- RGB → ab (Lab) ----------
def rgb_to_ab(r, g, b):
    rgb = np.uint8([[[r, g, b]]])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]
    return np.array([lab[1] - 128, lab[2] - 128], dtype=np.float32)

# ---------- PyQt5 Color Picker ----------
def open_color_picker_qt():
    global selected_rgb, selected_ab
    app = QApplication(sys.argv)
    color = QColorDialog.getColor()
    if color.isValid():
        r, g, b = color.red(), color.green(), color.blue()
        selected_rgb = (r, g, b)
        selected_ab = rgb_to_ab(r, g, b)
        print(f"Selected: RGB=({r},{g},{b}) → ab={selected_ab}")
    del app  # prevent QApplication conflict

# ---------- Argument Parser ----------
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='test_img/gray_apple.png', help='Path to grayscale image')
args = parser.parse_args()

# ---------- Load Image ----------
gray = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
assert gray is not None, f"Image not found: {args.img_path}"
img_for_drawing = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
h, w = gray.shape
ab_hint_map = np.zeros((h, w, 2), dtype=np.float32)
hint_mask = np.zeros((h, w), dtype=np.uint8)

# ---------- Mouse Scribble Callback ----------
def draw_scribble(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.circle(img_for_drawing, (x, y), 3, selected_rgb[::-1], -1)  # BGR for OpenCV
        ab_hint_map[y, x] = selected_ab
        hint_mask[y, x] = 1

# ---------- OpenCV GUI Loop ----------
cv2.namedWindow("Draw (p=color, s=save & run, q=quit)")
cv2.setMouseCallback("Draw (p=color, s=save & run, q=quit)", draw_scribble)

while True:
    cv2.imshow("Draw (p=color, s=save & run, q=quit)", img_for_drawing)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        open_color_picker_qt()
    elif key == ord('s'):
        np.save('ab_hint_map.npy', ab_hint_map)
        np.save('hint_mask.npy', hint_mask)
        print("Saved: ab_hint_map.npy and hint_mask.npy")

        # Automatically run model
        output_name = Path(args.img_path).stem
        os.makedirs("saved_outputs", exist_ok=True)
        os.system(f"python demo_ext.py --img_path {args.img_path} --save_prefix saved_outputs/{output_name}")
        break
    elif key == ord('q'):
        print("👋 Exiting.")
        break

cv2.destroyAllWindows()