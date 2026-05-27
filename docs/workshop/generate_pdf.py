#!/usr/bin/env python3
"""
Generate PDF from the workshop one-pager HTML.

Requirements:
    pip install weasyprint

Alternative (if weasyprint fails):
    pip install pdfkit
    # Also requires wkhtmltopdf: brew install wkhtmltopdf (Mac) or apt-get install wkhtmltopdf (Linux)
"""

import sys
import os
from pathlib import Path


def generate_pdf_weasyprint():
    """Generate PDF using WeasyPrint (better CSS support)."""
    try:
        from weasyprint import HTML

        html_file = Path(__file__).parent / "workshop-onepager.html"
        pdf_file = Path(__file__).parent / "workshop-onepager.pdf"

        print(f"Generating PDF from {html_file}...")
        HTML(filename=str(html_file)).write_pdf(str(pdf_file))
        print(f"✅ PDF generated: {pdf_file}")
        return True

    except ImportError:
        print("❌ WeasyPrint not installed.")
        return False
    except Exception as e:
        print(f"❌ Error with WeasyPrint: {e}")
        return False


def generate_pdf_pdfkit():
    """Generate PDF using pdfkit (requires wkhtmltopdf)."""
    try:
        import pdfkit

        html_file = Path(__file__).parent / "workshop-onepager.html"
        pdf_file = Path(__file__).parent / "workshop-onepager.pdf"

        # Options for better rendering
        options = {
            "page-size": "A4",
            "margin-top": "0.75in",
            "margin-right": "0.75in",
            "margin-bottom": "0.75in",
            "margin-left": "0.75in",
            "encoding": "UTF-8",
            "no-outline": None,
            "enable-local-file-access": None,
        }

        print(f"Generating PDF from {html_file}...")
        pdfkit.from_file(str(html_file), str(pdf_file), options=options)
        print(f"✅ PDF generated: {pdf_file}")
        return True

    except ImportError:
        print("❌ pdfkit not installed.")
        return False
    except Exception as e:
        print(f"❌ Error with pdfkit: {e}")
        return False


def print_manual_instructions():
    """Print instructions for manual PDF generation."""
    print("\n📄 Manual PDF Generation Options:")
    print("\n1. Using your browser:")
    print("   - Open workshop-onepager.html in Chrome/Firefox")
    print("   - Press Ctrl+P (or Cmd+P on Mac)")
    print("   - Choose 'Save as PDF'")
    print("   - Set margins to 'None' or 'Minimum'")
    print("   - Enable 'Background graphics'")
    print("   - Save as workshop-onepager.pdf")

    print("\n2. Using WeasyPrint (recommended):")
    print("   pip install weasyprint")
    print("   python generate_pdf.py")

    print("\n3. Using wkhtmltopdf:")
    print("   # Install wkhtmltopdf first:")
    print("   # Mac: brew install wkhtmltopdf")
    print("   # Linux: apt-get install wkhtmltopdf")
    print("   pip install pdfkit")
    print("   python generate_pdf.py")


def main():
    """Try different methods to generate PDF."""
    print("🖨️  Workshop One-Pager PDF Generator")
    print("=" * 40)

    # Try WeasyPrint first (better CSS support)
    if generate_pdf_weasyprint():
        return

    # Try pdfkit as fallback
    if generate_pdf_pdfkit():
        return

    # If both fail, show manual instructions
    print_manual_instructions()


if __name__ == "__main__":
    main()
