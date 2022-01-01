import cv2
import time

def ensure_gray(img):
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BRG2GRAY)
    return img

def clahe_equalize(img):
    
    img = ensure_gray(img)
    # start = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    # end = time.time()
    # print(f"Clahe in {end- start}s")
    return equalized

def standardize(img):
    
    img = img / 127.5
    img -= 1
    
    return img