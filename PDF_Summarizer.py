import streamlit as st
import PyPDF2
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Optional improvement for chunking

# Load the summarization model globally
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def extract_pdf_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

def split_text(text, max_chunk_size=1024):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size)
    return splitter.split_text(text)

# Streamlit app
def main():
    st.title("Cutting-Edge PDF Summarizer")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        text = extract_pdf_text(uploaded_file)
        if text.strip():
            # Display extracted text (optional)
            st.subheader("Extracted Text")
            st.text_area("Text from PDF", text, height=300)

            # Summarization options
            st.subheader("Processing the Summary")
            max_length = st.slider("Maximum length of the summary (words)", 50, 500, 150)
            min_length = st.slider("Minimum length of the summary (words)", 20, 100, 50)

            # Load summarizer
            summarizer = load_summarizer()

            # Split text into manageable chunks
            chunks = split_text(text)
            summary_list = []
            with st.spinner("Generating summary..."):
                for chunk in chunks:
                    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                    summary_list.append(summary[0]['summary_text'])

            # Display the summary
            st.subheader("Summary")
            st.write(" ".join(summary_list))
        else:
            st.warning("No text could be extracted from the uploaded PDF.")
    else:
        st.info("Please upload a PDF file to start.")

if __name__ == "__main__":
    main()
