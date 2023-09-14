from pdfminer.high_level import extract_text
import spacy
nlp = spacy.load('en_core_web_trf')
text=extract_text(r"C:\Users\VigneshSubramani\Pictures\Lease\Lease (7).pdf")
format_text = text.split()
clean_text = " ".join(format_text[0:6000])
doc=nlp(clean_text)
persons = []
organizations = []
for ent in doc.ents:
    if ent.label_ == 'PERSON':
        persons.append(ent.text)
    elif ent.label_ == 'ORG':
        organizations.append(ent.text)
# Print the extracted names
print("Persons:", persons)
print("Organizations:", organizations)