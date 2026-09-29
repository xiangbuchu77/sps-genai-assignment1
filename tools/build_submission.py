"""Build the submission PDF from the written solutions and saved API evidence.

Requires reportlab. This helper is separate from the API runtime dependencies.
"""

from pathlib import Path
import json
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def build(font_dir: Path):
    pdfmetrics.registerFont(TTFont("DV", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-Bold", italic="DV", boldItalic="DV-Bold")
    navy = colors.HexColor("#163A58")
    body = ParagraphStyle("body", fontName="DV", fontSize=9.5, leading=14, spaceAfter=7)
    heading = ParagraphStyle("heading", parent=body, fontName="DV-Bold", fontSize=13, leading=18, textColor=navy, spaceBefore=10, spaceAfter=9)
    title = ParagraphStyle("title", parent=heading, fontSize=24, leading=30, spaceAfter=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11)
    sub = ParagraphStyle("sub", parent=heading, fontSize=11, leading=15, spaceBefore=8, spaceAfter=6)
    mono = ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8.3, leading=12, backColor=colors.HexColor("#F0F4F7"), borderPadding=7, spaceBefore=5, spaceAfter=12)
    story = []

    def p(text, style=body):
        text = escape(text)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)
        story.append(Paragraph(text, style))

    p("Assignment 1", title)
    p("Coding Environment Setup and Simple API Implementation", heading)
    p("Applied Generative AI  |  Fall 2026", small)
    p("Part 1: FastAPI implementation", heading)
    url = "https://github.com/xiangbuchu77/sps-genai-assignment1"
    story.append(Paragraph(f'<b>GitHub repository:</b><br/><link href="{url}" color="#176CA4">{url}</link>', body))
    p("The project extends the Module 3 API with the spaCy word-embedding functionality from Module 2. It retains GET / and POST /generate and adds POST /embedding.")
    p("Embedding endpoint", sub)
    p('POST /embedding     Request: {"word": "apple"}', mono)
    p("The response contains word, model, dimensions, and the full embedding. The implementation uses en_core_web_lg 3.8.0 and the classroom operation nlp(input_word).vector. The model is loaded once at startup, and the NumPy vector is converted to a JSON list. Tagging and parsing components are excluded because this operation only needs tokenization and static vectors.")
    d = json.loads((ROOT / "evidence/embedding_apple.json").read_text())
    p(f"Observed response: word = {d['word']}; model = {d['model']}; dimensions = {d['dimensions']}.")
    p("First eight vector values (preview only): " + ", ".join(f"{x:.5f}" for x in d['embedding'][:8]), small)
    p("The complete 300-value response is saved in evidence/embedding_apple.json. Blank input, phrases, numbers, and punctuation return HTTP 422. An alphabetic token without a stored vector returns HTTP 404. Surrounding whitespace is removed; case is preserved.")
    p("Run and verify", sub)
    for line in ["uv sync --frozen", "uv run fastapi dev app/main.py", "uv run pytest -q", "uv run python probability_solutions.py"]:
        p(line, mono)
    p("Interactive documentation: http://127.0.0.1:8000/docs. The first installation downloads the model; requests then run locally. A Docker deployment is included and described on the next page.")
    v = json.loads((ROOT / "evidence/verification.json").read_text())
    p(f"Verification: {v['integration_tests']['passed']} integration tests passed. Eight live HTTP checks confirmed the root route, documentation, OpenAPI schema, embedding output, error responses, and text generation. Vector outputs matched the actual model exactly. Evidence and dependency versions are included in the repository.")

    story.append(PageBreak())
    p("Part 1: Docker deployment", heading)
    p("The repository includes Dockerfile, .dockerignore, pyproject.toml, and uv.lock. The image packages Python 3.12, the locked runtime dependencies, the application, and the en_core_web_lg model. The host only needs a running Docker installation; host Python and model files are not required.")
    p("1. Download the code and build", sub)
    for line in ["git clone https://github.com/xiangbuchu77/sps-genai-assignment1.git", "cd sps-genai-assignment1", "docker build -t sps-genai-assignment1 ."]:
        p(line, mono)
    p("The first build needs internet access for the base image, Python dependencies, and approximately 382 MiB model download. Allow several minutes and adequate disk space.")
    p("2. Start the container", sub)
    p("docker run --rm -d --name sps-genai-assignment1 \\", mono)
    p("  -p 127.0.0.1:8000:80 sps-genai-assignment1", mono)
    p("docker ps --filter name=sps-genai-assignment1", mono)
    p("Wait for the health status to become healthy. Open http://127.0.0.1:8000/docs to query the API. The server listens on 0.0.0.0:80 inside the container; Docker maps host port 8000 to container port 80. If port 8000 is occupied, use 127.0.0.1:8001:80 and open port 8001 instead.")
    p("3. Query the embedding endpoint", sub)
    for line in ["curl -X POST http://127.0.0.1:8000/embedding \\", "  -H 'Content-Type: application/json' \\", "  -d '{\"word\":\"apple\"}'"]:
        p(line, mono)
    p("Expected response: word = apple; model = en_core_web_lg; dimensions = 300; embedding = the full vector. GET / and POST /generate are also available in the container.")
    p("4. Logs and shutdown", sub)
    p("docker logs sps-genai-assignment1", mono)
    p("docker stop sps-genai-assignment1", mono)
    docker = json.loads((ROOT / "evidence/docker_verification.json").read_text())
    p("Container verification", sub)
    if docker['status'] == 'passed':
        p(f"A Docker image was built successfully and tested on {docker['platform']}. The running container reported healthy. All {len(docker['http_checks'])} live HTTP checks passed, including the full 300-dimensional embedding, validation errors, and text generation. The embedding matched the locally verified vector. The test used no host directory mounts. Reproduction details are recorded in evidence/docker_verification.json.")
    else:
        p("Docker build and run instructions are included. Local container execution has not yet been verified: Docker Desktop could not start because its virtual disk was owned by root and was not writable by the current user. This requires a host administrator to correct the file ownership. The Python API tests and local HTTP checks on the preceding page passed; they are not container tests.")

    text = (ROOT / "solutions.md").read_text()
    theory = text.split("## Part 2: Rules of Probability\n", 1)[1]
    story.append(PageBreak())
    p("Part 2: Rules of Probability", heading)
    lines = theory.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("### Question 5"):
            story.append(PageBreak())
        if line.startswith("### "):
            p(line[4:], sub)
        elif line.startswith("|"):
            rows=[]
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts=[x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", x) for x in parts):
                    rows.append([Paragraph(escape(x), small) for x in parts])
                i+=1
            table=Table(rows, colWidths=[65,105,140,170], hAlign="LEFT", repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E4EDF4")),
                ("LINEBELOW",(0,0),(-1,0),0.6,navy),
                ("LINEABOVE",(0,-1),(-1,-1),0.5,navy),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),6),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ]))
            story.extend([table,Spacer(1,10)])
            continue
        else:
            p(line)
        i+=1
    p("References", sub)
    p("Assignment1-1.pdf; Module 2 Practice 2: Word Sampling; Module 2 Practice 3: Word Embeddings; Module 3 Activity: First Docker/FastAPI Project Setup and Simple Text Generator.", small)
    for label,url in [("spaCy vectors and similarity","https://spacy.io/usage/linguistic-features#vectors-similarity"),("FastAPI testing","https://fastapi.tiangolo.com/tutorial/testing/"),("uv Docker integration","https://docs.astral.sh/uv/guides/integration/docker/")]:
        story.append(Paragraph(f'<link href="{url}" color="#176CA4">{label}</link>',small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8E1E7"))
        canvas.line(54,42,558,42)
        canvas.setFont("DV",8)
        canvas.setFillColor(colors.HexColor("#617482"))
        canvas.drawString(54,28,"Assignment 1 | Applied Generative AI")
        canvas.drawRightString(558,28,str(doc.page))
        canvas.restoreState()

    dest = ROOT / "output/pdf/Assignment1_Submission.pdf"
    dest.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(dest),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=40,bottomMargin=54,title="Assignment 1 - FastAPI and Probability",author="",allowSplitting=1).build(story,onFirstPage=footer,onLaterPages=footer)
    print(dest)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--font-dir",type=Path,required=True,help="Directory containing DejaVuSans.ttf and DejaVuSans-Bold.ttf")
    build(parser.parse_args().font_dir)
