import fitz  # PyMuPDF
import re

doc = fitz.open("9473_Improving_Transformers_wi.pdf")
def extract_impact_statement(doc):
  import fitz  # PyMuPDF
  import re
  test=0
  text_results = ""
  for page in doc:
    blocks = page.get_text("blocks")
    
    for block in blocks:
        text = block[4].strip()
        if(test):
            text_results+=text
            test-=1
        # vérifier si le bloc commence par "impact statement" (case insensitive)
        if re.match(r"^\s*impact statement", text, re.IGNORECASE):
            print("=== FOUND BLOCK ===")
            text_results+=text
            test+=1
            print()
  return text_results