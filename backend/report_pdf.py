"""
report_pdf.py
------------------------------------------------------------
Generates a real PDF file (bytes) for an AI Executive Report,
server-side, using ReportLab.

WHY THIS EXISTS:
The previous "Download PDF" implementation opened a blank browser
tab and called window.print(), relying on the user manually choosing
"Save as PDF" in the OS print dialog - not a real download, and it
opens a second page/tab. This module replaces that with an actual
PDF binary produced on the backend, so a single API call returns a
downloadable .pdf file directly (see /api/report/pdf in main.py).

INPUT FORMAT:
Report content is the markdown-ish text already produced by
copilot.generate_report() - a mix of:
    ## Heading
    plain paragraph text
    - bullet line
    * bullet line
    **bold text** (inline)
This module renders that into a clean, professional one-file PDF.
It intentionally supports a small, predictable subset of markdown
(headings, bullets, bold, paragraphs) rather than pulling in a full
HTML/markdown rendering stack - it only needs to handle what the AI
report prompts in copilot.py actually produce.
------------------------------------------------------------
"""

import io
import os
import re
import unicodedata
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable,
)

PRIMARY_COLOR = colors.HexColor("#0F6CBD")
TEXT_COLOR = colors.HexColor("#201F1E")
MUTED_COLOR = colors.HexColor("#5B5955")


# ---------------------------------------------------------------
# Unicode font registration
# ---------------------------------------------------------------
# ReportLab's built-in "Helvetica" is one of the 14 standard PDF core
# fonts - it only has a WinAnsi-style glyph set, so any character
# outside that (smart dashes, non-breaking hyphens, bullet glyphs,
# etc. - all things an LLM-generated report commonly emits) has no
# matching glyph and renders as a black "missing glyph" box (■)
# instead of raising an error.
#
# DejaVu Sans covers a much wider Unicode range, so we bundle it
# (backend/fonts/) and register it under the family name "DejaVuSans"
# for regular/bold/italic/bold-italic, then use it everywhere text is
# drawn below. Bundling the .ttf files (rather than pointing at a
# system font path) keeps this working regardless of what fonts are
# installed on the machine that ends up running the backend.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
UNICODE_FONT = "Helvetica"          # safe fallback if bundled fonts are missing
UNICODE_FONT_BOLD = "Helvetica-Bold"

try:
    pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVuSans",
        normal="DejaVuSans",
        bold="DejaVuSans-Bold",
        italic="DejaVuSans-Oblique",
        boldItalic="DejaVuSans-BoldOblique",
    )
    UNICODE_FONT = "DejaVuSans"
    UNICODE_FONT_BOLD = "DejaVuSans-Bold"
except Exception:
    # Bundled font files missing/unreadable - fall back to the core
    # Helvetica font rather than crashing report generation. Sanitizing
    # the text (below) still removes the most common problem characters
    # even in this fallback case.
    pass


# ---------------------------------------------------------------
# Text sanitization
# ---------------------------------------------------------------
# Maps specific "smart"/typographic characters (that a plain or even
# a Unicode font can render, but that look wrong or are a common
# source of confusion in a plain-text report) to safe, always-correct
# ASCII equivalents.
_CHAR_REPLACEMENTS = {
    "\u00A0": " ",   # non-breaking space
    "\u2011": "-",   # non-breaking hyphen (the actual cause of things
                      # like "30■day" - LLMs often use this inside
                      # compound words instead of a plain "-")
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2022": "-",   # bullet
    "\u25CF": "-",   # black circle bullet
    "\u25AA": "-",   # black small square bullet
    "\u2023": "-",   # triangular bullet
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201C": '"',   # left double quote
    "\u201D": '"',   # right double quote
    "\u2026": "...", # ellipsis
    "\u200B": "",    # zero-width space
    "\u200C": "",    # zero-width non-joiner
    "\u200D": "",    # zero-width joiner
    "\uFEFF": "",    # byte-order mark
}
_CHAR_REPLACEMENT_TABLE = str.maketrans(_CHAR_REPLACEMENTS)


def _sanitize_text(text: str) -> str:
    """Cleans report text before it reaches ReportLab so no character
    can render as a black missing-glyph box (■):
      1. Replace known "smart"/typographic characters with safe
         ASCII equivalents (non-breaking spaces, en/em dashes,
         non-breaking hyphens, bullet glyphs, smart quotes, etc).
      2. Strip control/formatting characters (e.g. stray zero-width
         or other Cf/Cc-category characters) that carry no visible
         meaning in a PDF report.
      3. Round-trip through UTF-8 so only well-formed, UTF-8-safe
         text reaches the PDF layer.
    Ordinary Unicode content (currency symbols, accented letters,
    punctuation, etc.) is left untouched - DejaVu Sans (registered
    above) already covers it, so there's no need to remove it.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)
    text = text.translate(_CHAR_REPLACEMENT_TABLE)

    # Drop remaining control/format characters (category "C*") other
    # than the whitespace ReportLab expects (\n, \t, plain space).
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t", " ") or not unicodedata.category(ch).startswith("C")
    )

    # Belt-and-suspenders: ensure the result is valid, UTF-8-encodable
    # text (drops anything that still can't round-trip cleanly).
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text


def _inline_markdown_to_html(text: str) -> str:
    """Converts a small set of inline markdown to ReportLab's mini-HTML
    (the only markup Paragraph() understands): **bold** and *italic*.
    Also escapes raw '&'/'<'/'>' first so user/AI content can't break
    the ReportLab markup parser."""
    text = _sanitize_text(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontSize=21, leading=25,
            textColor=PRIMARY_COLOR, spaceAfter=3, alignment=TA_LEFT,
            fontName=UNICODE_FONT_BOLD,
        ),
        "meta": ParagraphStyle(
            "ReportMeta", parent=base["Normal"], fontSize=9, leading=12,
            textColor=MUTED_COLOR, spaceAfter=18,
            fontName=UNICODE_FONT,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=base["Heading2"], fontSize=13.5, leading=17,
            textColor=TEXT_COLOR, spaceBefore=20, spaceAfter=8,
            fontName=UNICODE_FONT_BOLD,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["Normal"], fontSize=10, leading=15.5,
            textColor=TEXT_COLOR, spaceAfter=8,
            fontName=UNICODE_FONT,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet", parent=base["Normal"], fontSize=10, leading=15,
            textColor=TEXT_COLOR, spaceAfter=2,
            fontName=UNICODE_FONT, bulletFontName=UNICODE_FONT,
        ),
    }
    return styles


def _parse_report_lines(content: str, styles: dict):
    """Turns the AI-generated report text into a flat list of ReportLab
    flowables, grouping consecutive '- '/'* ' lines into one bullet list
    per section instead of one Flowable per line."""
    flowables = []
    pending_bullets = []

    def flush_bullets():
        if pending_bullets:
            flowables.append(
                ListFlowable(
                    [ListItem(Paragraph(_inline_markdown_to_html(b), styles["bullet"]), spaceAfter=4)
                     for b in pending_bullets],
                    bulletType="bullet", start="circle", leftIndent=16,
                )
            )
            flowables.append(Spacer(1, 6))
            pending_bullets.clear()

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line:
            flush_bullets()
            continue

        heading_match = re.match(r"^#{1,3}\s+(.*)$", line)
        if heading_match:
            flush_bullets()
            flowables.append(Paragraph(_inline_markdown_to_html(heading_match.group(1)), styles["h2"]))
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", line)
        if bullet_match:
            pending_bullets.append(bullet_match.group(1))
            continue

        numbered_match = re.match(r"^\d+[.)]\s+(.*)$", line)
        if numbered_match:
            pending_bullets.append(numbered_match.group(1))
            continue

        # Plain paragraph line
        flush_bullets()
        flowables.append(Paragraph(_inline_markdown_to_html(line), styles["body"]))

    flush_bullets()
    return flowables


def _draw_footer(canvas, doc):
    """Draws a subtle brand line + page number at the bottom of every page.
    Purely presentational (footer only) - never touches the report content
    or the page's usable content area beyond the existing bottom margin."""
    canvas.saveState()
    canvas.setFont(UNICODE_FONT, 8)
    canvas.setFillColor(MUTED_COLOR)
    canvas.drawString(0.85 * inch, 0.5 * inch, "SupplySphere AI \u00b7 Generated Report")
    canvas.drawRightString(LETTER[0] - 0.85 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#E1DFDD"))
    canvas.setLineWidth(0.5)
    canvas.line(0.85 * inch, 0.62 * inch, LETTER[0] - 0.85 * inch, 0.62 * inch)
    canvas.restoreState()


def build_report_pdf(title: str, content: str, generated_at: datetime = None) -> bytes:
    """
    Renders an executive report into a one-file PDF and returns the
    raw PDF bytes (ready to be sent as an HTTP response body with
    media_type="application/pdf").
    """
    generated_at = generated_at or datetime.now()
    styles = _build_styles()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.85 * inch, bottomMargin=0.9 * inch,
        title=title,
    )

    safe_title = _sanitize_text(title).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story = [
        Paragraph(safe_title, styles["title"]),
        Paragraph(f"SupplySphere AI &middot; Generated {generated_at.strftime('%d %b %Y, %H:%M')}", styles["meta"]),
        HRFlowable(width="100%", thickness=1.2, color=PRIMARY_COLOR, spaceAfter=10),
        Spacer(1, 4),
    ]
    story.extend(_parse_report_lines(content or "No report content available.", styles))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
