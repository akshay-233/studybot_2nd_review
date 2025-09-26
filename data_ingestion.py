import PyPDF2
from PyPDF2 import PdfReader
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


def count_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return len(reader.pages)
