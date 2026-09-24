import os
import csv
from typing import List, Dict

import pypdf
import pymupdf as fitz  # PyMuPDF: renders scanned PDF pages into images for OCR
from PIL import Image

# Tesseract OCR (optional: gracefully handle if not installed)
try:
    import pytesseract
    # Set the path to Tesseract on Windows
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class DocumentParser:
    """
    Multi-Modal document parser that extracts text from:
    PDFs (digital + scanned), Images (JPG/PNG), Excel, Word, CSV, and plain text.
    """
    # Map file extensions to their handler methods
    SUPPORTED_FORMATS = {
        ".pdf": "_parse_pdf",
        ".jpg": "_parse_image",
        ".jpeg": "_parse_image",
        ".png": "_parse_image",
        ".tiff": "_parse_image",
        ".tif": "_parse_image",
        ".xlsx": "_parse_excel",
        ".xls": "_parse_excel",
        ".docx": "_parse_word",
        ".csv": "_parse_csv",
        ".txt": "_parse_text",
    }

    def __init__(self):
        if OCR_AVAILABLE:
            print("DocumentParser initialized. OCR engine: Tesseract (Active).")
        else:
            print("DocumentParser initialized. OCR engine: Not installed (scanned documents will be flagged but not extracted).")

    def parse(self, file_path: str) -> List[Dict]:
        """
        Universal entry point. Detects file type and routes to the correct handler.

        Returns:
            List of dicts: [{"doc_name": "...", "page": 1, "text": "...", "is_scanned": False}]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        extension = os.path.splitext(file_name)[1].lower()

        if extension not in self.SUPPORTED_FORMATS:
            supported = ", ".join(self.SUPPORTED_FORMATS.keys())
            raise ValueError(f"Unsupported file format: '{extension}'. Supported: {supported}")

        # Dynamically call the correct handler method
        handler_name = self.SUPPORTED_FORMATS[extension]
        handler_method = getattr(self, handler_name)
        return handler_method(file_path, file_name)

    # ==================== PDF (Digital + Scanned OCR) ====================
    def _parse_pdf(self, file_path: str, file_name: str) -> List[Dict]:
        print(f"Parsing PDF: '{file_name}'...")
        reader = pypdf.PdfReader(file_path)
        total_pages = len(reader.pages)
        print(f"Found {total_pages} pages.")

        pages_data = []

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = " ".join(raw_text.split())

            is_scanned = len(cleaned_text.split()) < 10

            # If page appears scanned, attempt OCR using PyMuPDF to render page as image
            if is_scanned and OCR_AVAILABLE:
                print(f"  Page {page_number}: Low text detected. Running OCR...")
                cleaned_text = self._ocr_pdf_page(file_path, page_idx)
                is_scanned = len(cleaned_text.split()) < 10  # Re-check after OCR

            if is_scanned:
                print(f"  Page {page_number}: [Scanned/Empty] ({len(cleaned_text.split())} words extracted)")
            else:
                print(f"  Page {page_number}: [Digital] {len(cleaned_text)} chars, {len(cleaned_text.split())} words.")

            pages_data.append({
                "doc_name": file_name,
                "page": page_number,
                "text": cleaned_text,
                "is_scanned": is_scanned
            })

        return pages_data

    def _ocr_pdf_page(self, file_path: str, page_index: int) -> str:
        """Renders a single PDF page to an image and runs Tesseract OCR on it."""
        try:
            doc = fitz.open(file_path)
            page = doc[page_index]
            # Render page at 300 DPI for high-quality OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()

            text = pytesseract.image_to_string(img)
            return " ".join(text.split())
        except Exception as e:
            print(f"    OCR failed on page {page_index + 1}: {e}")
            return ""

    # ==================== Images (JPG, PNG, TIFF) ====================
    def _parse_image(self, file_path: str, file_name: str) -> List[Dict]:
        print(f"Parsing Image: '{file_name}'...")

        if not OCR_AVAILABLE:
            print("  [Error]: Tesseract OCR is not installed. Cannot extract text from images.")
            return [{"doc_name": file_name, "page": 1, "text": "", "is_scanned": True}]

        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        cleaned = " ".join(text.split())
        print(f"  OCR extracted {len(cleaned)} chars, {len(cleaned.split())} words.")

        return [{"doc_name": file_name, "page": 1, "text": cleaned, "is_scanned": True}]

    # ==================== Excel (.xlsx) ====================
    def _parse_excel(self, file_path: str, file_name: str) -> List[Dict]:
        import openpyxl
        print(f"Parsing Excel: '{file_name}'...")

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        pages_data = []

        for sheet_idx, sheet_name in enumerate(wb.sheetnames):
            sheet = wb[sheet_name]
            rows_text = []
            for row in sheet.iter_rows(values_only=True):
                # Convert each cell to string, skip None values
                cell_values = [str(cell) for cell in row if cell is not None]
                if cell_values:
                    rows_text.append(" | ".join(cell_values))

            full_text = " ".join(rows_text)
            print(f"  Sheet '{sheet_name}': {len(full_text)} chars, {len(rows_text)} rows.")

            pages_data.append({
                "doc_name": file_name,
                "page": sheet_idx + 1,
                "text": full_text,
                "is_scanned": False
            })

        wb.close()
        return pages_data

    # ==================== Word Documents (.docx) ====================
    def _parse_word(self, file_path: str, file_name: str) -> List[Dict]:
        import docx
        print(f"Parsing Word Document: '{file_name}'...")

        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = " ".join(paragraphs)
        print(f"  Extracted {len(paragraphs)} paragraphs, {len(full_text)} chars.")

        return [{"doc_name": file_name, "page": 1, "text": full_text, "is_scanned": False}]

    # ==================== CSV Files ====================
    def _parse_csv(self, file_path: str, file_name: str) -> List[Dict]:
        print(f"Parsing CSV: '{file_name}'...")

        rows_text = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                cell_values = [cell for cell in row if cell.strip()]
                if cell_values:
                    rows_text.append(" | ".join(cell_values))

        full_text = " ".join(rows_text)
        print(f"  Extracted {len(rows_text)} rows, {len(full_text)} chars.")

        return [{"doc_name": file_name, "page": 1, "text": full_text, "is_scanned": False}]

    # ==================== Plain Text (.txt) ====================
    def _parse_text(self, file_path: str, file_name: str) -> List[Dict]:
        print(f"Parsing Text File: '{file_name}'...")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = " ".join(f.read().split())

        print(f"  Extracted {len(full_text)} chars.")

        return [{"doc_name": file_name, "page": 1, "text": full_text, "is_scanned": False}]

    # ==================== Extraction Summary ====================
    def get_extraction_summary(self, pages_data: List[Dict]) -> Dict:
        total = len(pages_data)
        scanned = sum(1 for p in pages_data if p["is_scanned"])
        digital = total - scanned
        total_chars = sum(len(p["text"]) for p in pages_data)
        return {
            "total_pages": total,
            "digital_pages": digital,
            "scanned_pages": scanned,
            "total_characters": total_chars
        }


if __name__ == "__main__":
    # Smoke Test: Generate a real 2-page PDF and parse it
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as pdf_canvas

    test_pdf_path = "test_contract.pdf"
    c = pdf_canvas.Canvas(test_pdf_path, pagesize=letter)

    # Page 1
    c.setFont("Helvetica", 11)
    c.drawString(72, 720, "COMMERCIAL LEASE AGREEMENT")
    c.setFont("Helvetica", 9)
    y = 690
    for line in [
        "This agreement is between Landlord Corp LLC and Tenant Inc.",
        "The premises at 45 Innovation Drive, Nairobi shall be leased",
        "for twenty-four months commencing October 1, 2026.",
        "CLAUSE 1.1: Monthly rent is $3,500 USD payable on the first day.",
        "CLAUSE 1.2: Security deposit of $7,000 USD is required.",
    ]:
        c.drawString(72, y, line)
        y -= 14
    c.showPage()

    # Page 2
    c.setFont("Helvetica", 9)
    y = 720
    for line in [
        "CLAUSE 4.1: Late payments exceeding 5 days incur $250 penalty.",
        "CLAUSE 8.1: No pets of any kind permitted without written consent.",
        "CLAUSE 12.3: Security deposit returned within 14 business days.",
    ]:
        c.drawString(72, y, line)
        y -= 14
    c.showPage()
    c.save()
    print(f"Generated test PDF: '{test_pdf_path}'\n")

    # Test parsing
    parser = DocumentParser()
    pages = parser.parse(test_pdf_path)

    print("\n--- Extraction Summary ---")
    print(parser.get_extraction_summary(pages))

    print("\n--- Page Previews ---")
    for p in pages:
        preview = p["text"][:150] + "..." if len(p["text"]) > 150 else p["text"]
        print(f"  Page {p['page']}: \"{preview}\"")

    # Clean up
    os.remove(test_pdf_path)
    print(f"\nCleaned up: '{test_pdf_path}'")

    # Show all supported formats
    print(f"\nSupported formats: {list(DocumentParser.SUPPORTED_FORMATS.keys())}")