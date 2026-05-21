# Invoice Validation System

An AI-powered Invoice Validation and OCR Accuracy Testing System built using Python, Streamlit, EasyOCR, OpenCV, and Machine Learning.  
The application automates invoice text extraction, validates extracted fields, and generates accuracy reports for OCR performance evaluation.

---

# Features

- OCR-based invoice text extraction
- Automatic Vendor, Date, and Total detection
- Batch invoice processing using ZIP uploads
- Accuracy validation dashboard
- Validation report generation
- Machine Learning pipeline for intelligent extraction
- Interactive Streamlit web interface

---

# Tech Stack

## Frontend / Dashboard
- Streamlit

## OCR & Image Processing
- EasyOCR
- OpenCV
- Pillow

## Machine Learning
- Scikit-learn
- RandomForestClassifier
- TF-IDF Vectorization

## Data Processing
- Pandas
- NumPy

---

# Project Architecture

```text
User Uploads ZIP Invoices
            │
            ▼
      Image Preprocessing
            │
            ▼
       OCR Text Extraction
            │
            ▼
    Vendor / Date / Total Extraction
            │
            ▼
      ML Validation Pipeline
            │
            ▼
      Accuracy Report Dashboard
```

---

# Project Structure

```bash
invoice-validation-system/
│
├── app.py
├── train_model.py
├── validator.py
├── requirements.txt
├── invoice_model.pkl
├── training_data.csv
│
├── sample_invoices/
│
├── screenshots/
│   ├── dashboard.png
│   ├── reports.png
│
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/invoice-validation-system.git
cd invoice-validation-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

The application will start locally in your browser.

---

# Machine Learning Pipeline

The project uses:

- TF-IDF Vectorization for text feature extraction
- MultiOutputClassifier for multi-field prediction
- RandomForestClassifier for robust classification

Model training is handled in:

```bash
train_model.py
```

To retrain the model:

```bash
python train_model.py
```

---

# OCR Workflow

1. Upload invoice images in ZIP format
2. Images are preprocessed using OpenCV
3. EasyOCR extracts invoice text
4. Regex + ML extract:
   - Vendor
   - Invoice Date
   - Total Amount
5. Validation metrics are generated

---

# Validation Metrics

The system evaluates:

- Vendor extraction accuracy
- Date extraction accuracy
- Total amount accuracy
- Precision & recall metrics

---

# Screenshots

## Dashboard

Add screenshot here:

```text
screenshots/dashboard.png
```

## Reports

Add screenshot here:

```text
screenshots/reports.png
```

---

# Example Use Cases

- Invoice automation systems
- OCR testing pipelines
- Finance document validation
- AI-based document processing
- Accounts payable automation

---

# Future Improvements

- PDF invoice support
- Deep learning OCR models
- Confidence score visualization
- Cloud deployment (AWS/Azure)
- Database integration
- REST API support
- Docker containerization
- Real-time invoice validation

---

# Requirements

Main dependencies:

```text
streamlit
pandas
easyocr
opencv-python
numpy
Pillow
scikit-learn
joblib
```

---

