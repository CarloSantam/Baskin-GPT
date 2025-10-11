import fitz  # PyMuPDF
import re

# --- funzione per pulire testo ---
def clean_text(text: str) -> str:
    # Remove multiple spaces
    text = re.sub(r'Regolamento di Gioco\s*–\s*Disciplina Baskin\s*\(Rev\.\s*\d+\)\s*Pag\.\s*\d+', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'Regolamento di Gioco\s*–\s*Disciplina Baskin\s*\(Rev\.\s*\d+\)', '', text, flags=re.IGNORECASE)


    text = re.sub(r'\s+', ' ', text)
    
    # Remove "Page X" or "Pagina X"
    text = re.sub(r'\b(Page|Pagina)\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove isolated numbers (es. numeri di pagina)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    text = re.sub(r"[•●◦▪▫◘◙]", "", text)

    
    # Trim spaces
    return text.strip()

# --- apri PDF ---
pdf_file = "Regolamento-di-Gioco-Disciplina-Baskin-Rev.-20.pdf"
doc_pdf = fitz.open(pdf_file)

# --- estrai e pulisci testo ---
all_text = ""
for page in doc_pdf:
    raw_text = page.get_text("text")
    text = clean_text(raw_text)
    if text:
        all_text += text + "\n\n"  # separa le pagine con newline

# --- salva in TXT ---
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("✅ Conversione completata: 'output.txt'")

######################################################

# import json

# with open('output.txt') as f:
#     testo=f.read()
    
# frasi = [frase.strip() for frase in testo.split(".") if frase.strip()]
# print(frasi)

# dati=[{'sentence':frase,'categoria':'Baskin'} for frase in frasi]

# with open("baskin_regolamento.json", "w", encoding="utf-8") as f:
#     json.dump(dati, f, ensure_ascii=False)


import json
import re

# Full regulation text from user

with open('output.txt') as f:
    testo=f.read()
    
# Regex to split into sentences more accurately
pattern = r'(?<=[.!?])\s+(?=[A-ZÀ-Ú])|(?=\bRegola\s+\d)|(?=\bFigura\s+\d)|(?=\bnumero\s+\d)|(?=\bDeroga\s+di)'
sentences = re.split(pattern, testo)

# Clean up sentences
cleaned_sentences = [s.strip() for s in sentences if s.strip() and s.strip() != "Mt."]

# Build JSON
data_detailed = [{"sentence": s, "categoria": "Baskin"} for s in cleaned_sentences]

# Save to JSON
file_path_detailed = "baskin_regolamento.json"
with open(file_path_detailed, "w", encoding="utf-8") as f:
    json.dump(data_detailed, f, ensure_ascii=False, indent=4)

len(data_detailed), file_path_detailed