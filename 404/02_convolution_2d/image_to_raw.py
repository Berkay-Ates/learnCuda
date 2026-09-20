#!/usr/bin/env python3
"""JPEG -> grayscale raw uint8 dosyasi. CUDA tarafinin okuyacagi format:
sadece width*height adet ham byte, hicbir header yok (boyutlar .cu icinde
sabit tanimli, bu script ile eslesmeli)."""
import sys
from PIL import Image

src = sys.argv[1] if len(sys.argv) > 1 else "lena.jpeg"
dst = sys.argv[2] if len(sys.argv) > 2 else "lena_gray.raw"

img = Image.open(src).convert("L")  # "L" = grayscale (0-255 tek kanal)
print(f"Boyut: {img.width}x{img.height}")
with open(dst, "wb") as f:
    f.write(img.tobytes())
print(f"Yazildi: {dst} ({img.width * img.height} byte)")
