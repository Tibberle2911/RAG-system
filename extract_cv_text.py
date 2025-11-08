import sys
from pathlib import Path

def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader  # pypdf is lightweight successor of PyPDF2
    except ImportError:
        raise SystemExit("pypdf not installed. Run: pip install pypdf")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = f"[Page {i+1} extraction error: {e}]"
        text_parts.append(f"\n===== PAGE {i+1} =====\n{text.strip()}\n")
    return "".join(text_parts)

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_cv_text.py <pdf_path> <output_txt>")
        sys.exit(1)
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = extract_pdf_text(pdf)
    out.write_text(text, encoding="utf-8")
    print(f"Extracted text written to {out}")

if __name__ == "__main__":
    main()