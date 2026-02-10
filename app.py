import os
from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def summarize(text):
    text = text[:5000]

    prompt = f"""
You are a financial research assistant.

Based ONLY on the text below, produce:

Management Tone:
Confidence Level:

Key Positives:
1.
2.
3.

Key Concerns:
1.
2.
3.

Forward Guidance:
- Revenue:
- Margin:
- Capex:

Capacity Utilization Trends:

Growth Initiatives:
1.
2.

Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        file = request.files["file"]

        try:
            text = extract_text(file)

            if not text.strip():
                error = "Could not extract text."
            else:
                result = summarize(text)

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
