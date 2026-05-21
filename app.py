import streamlit as st
import pandas as pd
import easyocr
import numpy as np
import cv2
import zipfile
import os
import re
import datetime
from PIL import Image

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Invoice Validation", layout="wide")

# -----------------------------------
# HEADER
# -----------------------------------
c1, c2 = st.columns([0.8, 0.2])
with c1:
    st.title("Invoice Validation")
    st.caption("OCR/ML Accuracy Testing")
with c2:
    st.write("Welcome, **krish**")
    st.button("Logout")

st.header("Test Case Management")
st.write("Create test cases, validate extraction accuracy, and generate detailed reports.")

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "data" not in st.session_state:
    st.session_state.data = []

if "upload_folder" not in st.session_state:
    st.session_state.upload_folder = None

# -----------------------------------
# LOAD OCR
# -----------------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# -----------------------------------
# FUNCTIONS
# -----------------------------------
def preprocess_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    return gray

def ocr_extract(image_path):
    image = Image.open(image_path)
    img_array = np.array(image)
    processed = preprocess_image(img_array)
    results = reader.readtext(processed, detail=1, paragraph=True)
    text = "\n".join([res[1] for res in results])
    return text

def extract_fields(text):
    lines = text.split("\n")

    vendor = lines[0].strip() if lines else ""

    date_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
    date = date_match.group() if date_match else ""

    total_match = re.search(r'Total[:\s]*([\d\.]+)', text, re.IGNORECASE)
    total = total_match.group(1) if total_match else ""

    return vendor, date, total

# -----------------------------------
# TABS
# -----------------------------------
tab_create, tab_cases, tab_reports = st.tabs(["📝 Create", "📋 Test Cases", "📊 Reports"])

# ===================================
# CREATE TAB
# ===================================
with tab_create:

    st.subheader("Create Test Case")

    test_name = st.text_input("Test Name")
    notes = st.text_input("Notes (Optional)")

    st.write("---")
    st.subheader("📂 Upload Invoice Folder (ZIP)")

    zip_file = st.file_uploader("Upload ZIP file", type=["zip"])

    if zip_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_folder = f"uploaded_invoices_{timestamp}"
        os.makedirs(upload_folder, exist_ok=True)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(upload_folder)

        st.session_state.upload_folder = upload_folder
        st.success("Invoices extracted successfully!")

    if st.button("🚀 Process All Invoices"):

        folder = st.session_state.upload_folder

        if not folder:
            st.warning("Upload ZIP first")
        else:
            results_list = []

            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        path = os.path.join(root, file)

                        text = ocr_extract(path)
                        vendor, date, total = extract_fields(text)

                        entry = {
                            "name": file,
                            "notes": notes,
                            "vendor_exp": "",
                            "vendor_ext": vendor,
                            "date_exp": "",
                            "date_ext": date,
                            "total_exp": "",
                            "total_ext": total,
                        }

                        results_list.append(entry)

            st.session_state.data.extend(results_list)
            st.success(f"{len(results_list)} invoices processed!")

# ===================================
# TEST CASES TAB
# ===================================
with tab_cases:

    st.subheader("Test Cases")

    if not st.session_state.data:
        st.info("No test cases yet.")
    else:
        df = pd.DataFrame(st.session_state.data)
        st.dataframe(df, width="stretch")

# ===================================
# REPORTS TAB
# ===================================
with tab_reports:

    st.subheader("Validation Report")

    if not st.session_state.data:
        st.info("No test cases to evaluate.")
    else:
        df = pd.DataFrame(st.session_state.data)

        total_cases = len(df)
        vendor_matches = (df["vendor_exp"] == df["vendor_ext"]).sum()
        date_matches = (df["date_exp"] == df["date_ext"]).sum()
        total_matches = (df["total_exp"] == df["total_ext"]).sum()

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Cases Evaluated", total_cases)
        m2.metric("Vendor Accuracy", f"{(vendor_matches/total_cases)*100:.1f}%")
        m3.metric("Date Accuracy", f"{(date_matches/total_cases)*100:.1f}%")
        m4.metric("Total Accuracy", f"{(total_matches/total_cases)*100:.1f}%")

        st.write("---")
        st.dataframe(df, width="stretch")