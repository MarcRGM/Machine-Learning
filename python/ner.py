import spacy

# English NER model
ner = spacy.load("en_core_web_sm") 

entities = ner(input("Enter your sentence: "))

for ent in entities.ents:
    print(ent.text, "->", ent.label_)

