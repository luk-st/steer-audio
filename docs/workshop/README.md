# AI Research Assistant Workshop Materials

This directory contains all materials for the AI Research Assistant Workshop.

## 📚 Workshop Documents

### For Participants
- **[participant-guide.md](participant-guide.md)** - Complete 3-4 hour hands-on workshop guide
- **[quick-reference.md](quick-reference.md)** - Essential commands and patterns reference card
- **[troubleshooting.md](troubleshooting.md)** - Common issues and solutions

### For Facilitators
- **[facilitator-guide.md](facilitator-guide.md)** - Detailed guide for running the workshop

### One-Pager Summary
- **[workshop-onepager.md](workshop-onepager.md)** - Concise terminal-style overview (Markdown)
- **[workshop-onepager.html](workshop-onepager.html)** - Interactive terminal-themed version (HTML)
- **workshop-onepager.pdf** - Printable version (generate using instructions below)

## 🖨️ Generating the PDF One-Pager

### Option 1: Using Your Browser (Easiest)
1. Open `workshop-onepager.html` in Chrome or Firefox
2. Press `Ctrl+P` (or `Cmd+P` on Mac)
3. Settings:
   - Destination: "Save as PDF"
   - Margins: "None" or "Minimum"
   - ✓ Enable "Background graphics"
4. Click "Save" and name it `workshop-onepager.pdf`

### Option 2: Using the Python Script
```bash
# First, install dependencies
pip install weasyprint

# Then run the generator
python generate_pdf.py
```

If WeasyPrint doesn't work, try:
```bash
# Mac
brew install wkhtmltopdf
pip install pdfkit

# Linux
sudo apt-get install wkhtmltopdf
pip install pdfkit

# Then run
python generate_pdf.py
```

## 🎯 Workshop Structure

The workshop follows a 7-part progressive structure:

1. **Get Comfortable with Claude Code** (20 min)
2. **Gather Context for Your Codebase** (30 min)
3. **Speech-to-Text Project Outline** (20 min)
4. **Create Your Ideal CLAUDE.md** (20 min)
5. **Automate an Existing Task** (45 min)
6. **Implement Something New** (45 min)
7. **Observability & Verification** (20 min, optional)

## 🚀 Quick Start for Participants

1. Clone the template repository
2. Follow the participant guide starting with Part 1
3. Use the quick reference card for command reminders
4. Refer to troubleshooting guide when stuck

## 👨‍🏫 Quick Start for Facilitators

1. Review the facilitator guide thoroughly
2. Test the full workshop flow yourself
3. Prepare example outputs for each section
4. Set up communication channels for participants
5. Have the one-pager ready to share

## 📝 Customization

Feel free to adapt these materials for your specific research domain or institutional needs. The materials are designed to be modular - you can extend or shorten sections based on your workshop duration and participant experience level.