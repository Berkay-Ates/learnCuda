#!/usr/bin/env python3
"""CUDA'nin urettigi ham grayscale byte dosyasini goruntulenebilir bir PNG'ye
cevirir. Kullanim: raw_to_image.py <raw_dosya> <width> <height> <cikti.png>"""
import sys
from PIL import Image

src = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
dst = sys.argv[4]

with open(src, "rb") as f:
    data = f.read()

expected = width * height
if len(data) != expected:
    print(f"UYARI: dosya {len(data)} byte, {expected} bekleniyordu ({width}x{height})")

img = Image.frombytes("L", (width, height), data)
img.save(dst)
print(f"Yazildi: {dst}")
