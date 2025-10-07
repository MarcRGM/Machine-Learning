import spacy

# English NER model
nlp = spacy.load("en_core_web_sm") 

sentence = nlp(input("Enter your sentence: "))

for token in sentence:
    print(f"{token.text:<10} {token.pos_:<10} {token.tag_:<10} {spacy.explain(token.tag_)}")

