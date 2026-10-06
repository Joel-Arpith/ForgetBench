"""Fill the report's Table of Contents with real entries and page numbers.

The docx TOC is a Word field that only Word fills in, so other viewers (Google
Docs, Preview, the Claude file pane) show it empty. This renders the docx to PDF
with LibreOffice, reads which printed page each heading lands on, and writes
those entries into the field's cached result. Word still refreshes it on open.

usage: python fill_toc.py report.docx   (needs LibreOffice + pymupdf)
"""
import re, subprocess, sys, tempfile, zipfile, shutil, os
from xml.sax.saxutils import escape
import fitz

SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()


def headings(xml):
    out = []
    for p in re.findall(r"<w:p>.*?</w:p>|<w:p .*?</w:p>", xml, flags=re.S):
        m = re.search(r'<w:pStyle w:val="Heading([123])"/>', p)
        if m:
            out.append((int(m.group(1)), "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))))
    return out


def page_labels(pdf):
    doc = fitz.open(pdf)
    pages = []
    for pg in doc:
        blocks = [b for b in pg.get_text("blocks") if b[4].strip()]
        foot = max(blocks, key=lambda b: b[1])[4].strip() if blocks else ""
        label = foot if re.fullmatch(r"[0-9]+|[ivxlcdm]+", foot) else ""
        pages.append((label, norm(pg.get_text())))
    return pages


def main(path):
    tmp = tempfile.mkdtemp()
    subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", tmp, path],
                   check=True, capture_output=True)
    pages = page_labels(os.path.join(tmp, os.path.splitext(os.path.basename(path))[0] + ".pdf"))

    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode()
    heads = headings(xml)

    # find each heading on or after the previous one's page; skip the TOC/lists pages themselves
    start = next(i for i, (_, t) in enumerate(pages) if "list of figures" in t) + 1
    entries, i = [], start
    for lvl, text in heads:
        key = norm(text)
        j = next((k for k in range(i, len(pages)) if key in pages[k][1]), None)
        assert j is not None, f"heading not found in PDF: {text}"
        entries.append((lvl, text, pages[j][0])); i = j
    assert all(e[2] for e in entries), "a heading landed on a page without a number"

    w = int(re.search(r'<w:pgSz w:w="(\d+)"', xml).group(1))
    ml = int(re.search(r'w:left="(\d+)"', re.search(r"<w:pgMar[^>]*/>", xml).group(0)).group(1))
    mr = int(re.search(r'w:right="(\d+)"', re.search(r"<w:pgMar[^>]*/>", xml).group(0)).group(1))
    tab = w - ml - mr

    def para(lvl, text, pg, pre="", post=""):
        b = "<w:b/>" if lvl == 1 else ""
        return (f'<w:p><w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{tab}"/></w:tabs>'
                f'<w:spacing w:before="{120 if lvl == 1 else 0}" w:after="60"/><w:ind w:left="{(lvl - 1) * 440}"/></w:pPr>'
                f'{pre}<w:r><w:rPr>{b}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
                f'<w:r><w:rPr>{b}</w:rPr><w:tab/></w:r><w:r><w:rPr>{b}</w:rPr><w:t>{pg}</w:t></w:r>{post}</w:p>')

    m = re.search(r'<w:p><w:r><w:fldChar w:fldCharType="begin"[^>]*/><w:instrText xml:space="preserve">TOC [^<]*</w:instrText>'
                  r'<w:fldChar w:fldCharType="separate"/></w:r></w:p><w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>', xml)
    assert m, "empty TOC field not found (already filled?)"
    begin = m.group(0).split("</w:r></w:p>")[0][len("<w:p>"):] + "</w:r>"
    body = [para(l, t, p) for l, t, p in entries]
    body[0] = para(*entries[0], pre=begin)
    body[-1] = para(*entries[-1], post='<w:r><w:fldChar w:fldCharType="end"/></w:r>')
    xml = xml[:m.start()] + "".join(body) + xml[m.end():]

    out = path + ".tmp"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as o:
        for item in z.infolist():
            o.writestr(item, xml if item.filename == "word/document.xml" else z.read(item.filename))
    z.close(); shutil.move(out, path); shutil.rmtree(tmp)
    for l, t, p in entries:
        print(f"{'  ' * (l - 1)}{t} .... {p}")


if __name__ == "__main__":
    main(sys.argv[1])
