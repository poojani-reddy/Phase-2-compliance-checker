import streamlit as st
from similarity_search import vec, Synthesizer
import base64
import PyPDF2
import pandas as pd
from fpdf import FPDF
import logging
import zipfile
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper function to chunk text
def chunk_text(text, chunk_size=8000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Helper function to read PDF file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Helper function to read TXT file
def read_txt(file):
    return file.getvalue().decode("utf-8")

# Generate download link for a single PDF file
def get_pdf_download_link(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(pdf_path)}">Download PDF Report</a>'
        return href

# Generate download link for a ZIP file
def get_zip_download_link(zip_path):
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
        b64 = base64.b64encode(zip_bytes).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{os.path.basename(zip_path)}">Download All Reports</a>'
        return href

# Process large text in chunks
def process_large_text(text):
    chunks = chunk_text(text)
    all_results = []

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)

        results = vec.search(chunk, limit=3)
        if not results.empty:
            all_results.append(results)

    if not all_results:
        return pd.DataFrame()

    combined_results = pd.concat(all_results, ignore_index=True)

    if 'content' not in combined_results.columns:
        raise ValueError("Expected 'content' column not found in search results.")

    combined_results = combined_results.drop_duplicates(subset=['content'])

    if {'agreement_date', 'effective_date', 'expiration_date'}.issubset(combined_results.columns):
        combined_results['metadata'] = combined_results.apply(
            lambda x: {
                'agreement_date': x['agreement_date'],
                'effective_date': x['effective_date'],
                'expiration_date': x['expiration_date']
            },
            axis=1
        )
    else:
        combined_results['metadata'] = [{} for _ in range(len(combined_results))]

    return combined_results

# Generate PDF report
def create_pdf_report(response, filename, no_results=False):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Contract Analysis Report", ln=True, align='C')
    pdf.ln(10)

    if no_results:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "No relevant results found in the contract.", ln=True)
        pdf.set_font("Helvetica", "B", 11)
        pdf.ln(5)
        pdf.multi_cell(0, 7, "Please review the contract content or embeddings.")
    else:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Analysis Summary:", ln=True)
        pdf.set_font("Helvetica", "B", 11)
        for para in response.answer.split('\n'):
            cleaned_para = para.strip()
            if cleaned_para:
                pdf.multi_cell(w=0, h=7, text=cleaned_para)
                pdf.ln(3)

    pdf.output(filename)

# Main Streamlit App
def main():
    st.title("Multi-Contract Analysis System")

    uploaded_files = st.file_uploader("Upload Contract Documents (PDF or TXT)", type=['pdf', 'txt'], accept_multiple_files=True)

    if uploaded_files:
        reports_dir = "contract_reports"
        os.makedirs(reports_dir, exist_ok=True)

        zip_file_path = os.path.join(reports_dir, "all_reports.zip")

        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for uploaded_file in uploaded_files:
                try:
                    # Read the uploaded file
                    if uploaded_file.type == "application/pdf":
                        contract_text = read_pdf(uploaded_file)
                        st.success(f"PDF file '{uploaded_file.name}' uploaded successfully!")
                    else:
                        contract_text = read_txt(uploaded_file)
                        st.success(f"TXT file '{uploaded_file.name}' uploaded successfully!")

                    # Display extracted text
                    with st.expander(f"View Extracted Text ({uploaded_file.name})"):
                        st.text_area(f"Contract Text - {uploaded_file.name}", contract_text, height=200)

                    # Analyze contract
                    results = process_large_text(contract_text)

                    report_filename = os.path.join(reports_dir, f"{os.path.splitext(uploaded_file.name)[0]}_analysis_report.pdf")

                    if results.empty:
                        create_pdf_report(None, report_filename, no_results=True)
                        st.warning(f"No relevant results found for '{uploaded_file.name}'.")
                    else:
                        response = Synthesizer.generate_response(
                            question=contract_text,
                            context=results[['content', 'metadata']]
                        )
                        create_pdf_report(response, report_filename)

                    zipf.write(report_filename, os.path.basename(report_filename))

                except Exception as e:
                    st.error(f"Error processing file '{uploaded_file.name}': {str(e)}")

        st.success("All contracts have been processed.")
        st.markdown(get_zip_download_link(zip_file_path), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
