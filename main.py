import pytesseract as pt
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# plate = pt.image_to_string(threshold, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_image(image):
    img = Image.open(image)
    return img


def preprocess_image(image):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 3)
    return img


def recognize_text(image):
    text = pt.image_to_string(image, lang='eng+rus')
    return text


def show_results(text):
    st.write("Распознанный текст:")
    st.write(text)


def main():
    st.title("Распознавание текста с картинки")
    uploaded_image = st.file_uploader("Загрузите изображение", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        image = load_image(uploaded_image)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        processed_image = preprocess_image(image)
        text = recognize_text(processed_image)
        show_results(text)


if __name__ == '__main__':
    main()
