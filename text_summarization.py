import fitz  # PyMuPDF
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import textwrap 

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def text_summarizer_from_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)

    # Load BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + pdf_text, return_tensors="pt", max_length=8048, truncation=True)
    summary_ids = model.generate(inputs, max_length=550, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "\n".join(textwrap.wrap(summary, width=80))  # Adjust width as needed
    return formatted_summary

def save_summary_as_pdf(pdf_path, summary):
    # Create a PDF document
    doc = fitz.open()

    page = doc.new_page()
    page.insert_text((20, 50), summary, fontname="helv", fontsize=15)

    output_pdf_path = pdf_path.replace(".pdf", "_summary.pdf")
    doc.save(output_pdf_path)
    doc.close()

    return output_pdf_path


pdf_file_path = "02.intro.news.articles.pdf"
summary = text_summarizer_from_pdf(pdf_file_path)
output_pdf_path = save_summary_as_pdf(pdf_file_path, summary)
print("Summary saved as PDF:", output_pdf_path)