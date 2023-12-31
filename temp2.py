# import language as language

import numpy as np
import pytesseract
import streamlit as st
# from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError,
#                                 PDFPopplerTimeoutError, PDFSyntaxError)

import helpers.constants as constants
import helpers.opencv as opencv
import helpers.pdfimage as pdfimage
import helpers.tesseract as tesseract
import helpers.easy_ocr as easy_ocr

pytesseract.pytesseract.tesseract_cmd = None

@st.cache_resource
def set_tesseract_path(tesseract_path: str):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
# plate = pt.image_to_string(threshold, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')
# pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


psm = st.selectbox(label="Page segmentation mode", options=constants.psm, index=3)
oem_index = 3
psm_index = constants.psm.index(psm)
language = st.selectbox(label="Select Language", options=list(constants.languages_sorted.values()),
                        index=constants.default_language_index)
language_short = list(constants.languages_sorted.keys())[list(constants.languages_sorted.values()).index(language)]
custom_oem_psm_config = tesseract.get_tesseract_config(oem_index=oem_index, psm_index=psm_index)


def load_image(uploaded_file):
    image = opencv.load_image(uploaded_file)
    return image


def preprocess_image(image):
    img = opencv.convert_to_rgb(image)
    return img


def recognize_text(image):
    text = pytesseract.image_to_string(image=image,
                              lang=language_short,
                              output_type=pytesseract.Output.STRING,
                              config=custom_oem_psm_config)
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
