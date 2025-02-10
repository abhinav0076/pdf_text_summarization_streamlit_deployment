import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from transformers import pipeline
import tempfile

# Initialize the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(file):
    """Extract text from PDF using PyMuPDF."""
    text = ""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_images(file):
    """Extract text from images in a PDF using OCR."""
    text = ""
    try:
        images = convert_from_bytes(file.read())
        for image in images:
            text += pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error during OCR: {e}")
    return text

def summarize_text(text):
    """Summarize the extracted text."""
    try:
        # BART has a max token limit; split text if too long
        max_chunk_size = 1024  # Adjust based on model's max input size
        text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in text_chunks]
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

def main():
    st.title("PDF Summarizer")
    st.write("Upload a PDF file, and the app will extract and summarize its text content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Attempt to extract text directly
            extracted_text = extract_text_from_pdf(uploaded_file)

            if not extracted_text:
                st.warning("No text found in PDF; attempting OCR.")
                # Reset file pointer and attempt OCR
                uploaded_file.seek(0)
                extracted_text = extract_text_from_images(uploaded_file)

            if extracted_text:
                st.subheader("Extracted Text")
                st.write(extracted_text[:2000])  # Display first 2000 characters

                summary = summarize_text(extracted_text)
                if summary:
                    st.subheader("Summary")
                    st.write(summary)
                else:
                    st.error("Summarization failed.")
            else:
                st.error("Failed to extract text from the PDF.")

if __name__ == "__main__":
    main()
