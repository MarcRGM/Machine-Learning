import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# spaCy - performs Named Entity Recognition (NER) — detects people, organizations, and places in text.
# pandas - used to store and sort results easily.
# matplotlib.pyplot - used to create the bar chart visualization.
# collections.Counter - counts how many times each entity appears across different articles.

# Load English NER model
nlp = spacy.load("en_core_web_sm")
# "en_core_web_sm" = small English model, fast and good for testing.
# The variable nlp acts as a processor — you pass text into it, and it returns a “Doc” object containing linguistic info.

# Example news articles 
articles = [
    "Apple announced the new iPhone 16 during its event in California.",
    "Samsung is preparing to compete with Apple's new device this year.",
    "Elon Musk visited Tesla’s Gigafactory in Berlin and met with German officials.",
    "Apple and Microsoft are investing in artificial intelligence startups.",
    "Barack Obama praised Tesla's innovation in electric cars.",
    "Microsoft unveiled new AI tools for developers during its Build conference.",
    "Tesla’s shares rose after Apple revealed its new battery partnership.",
    "Google and Microsoft are leading the race in AI development.",
    "Apple's CEO Tim Cook said innovation will continue."
]

# Counter to track how many articles each entity appears in
entity_counter = Counter()

# Process each article one by one
for text in articles:
    doc = nlp(text)  # Analyze the text using spaCy's model

    # Use a set to ensure each entity is only counted once per article
    entities_in_article = set()

    for ent in doc.ents: 
        if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Keep people, organizations, and places
            entities_in_article.add(ent.text)

    # Increment count by 1 for each entity found in this article
    for entity in entities_in_article:
        entity_counter[entity] += 1

# Convert results to a DataFrame
df = pd.DataFrame(entity_counter.items(), columns=["Entity", "Articles Mentioned In"])
# entity_counter.items() returns all entity–count pairs from Counter() in this form: 
# [("Apple", 3), ("Microsoft", 2), ("Tesla", 2), ...].
# pd.DataFrame(...) converts that list into a table-like structure (DataFrame). 
# columns=["Entity", "Articles Mentioned In"] names the two columns: 
# first is the entity name, second is the number of articles that mentioned it.
df = df.sort_values(by="Articles Mentioned In", ascending=False)
# sort_values() rearranges rows in a DataFrame.
# ascending=False shows highest first — so the most trending entities appear on top.

# Show results in the console
print("\nTop Trending Entities (by number of articles mentioned in):\n")
print(df)

# Plot trending entities
plt.figure(figsize=(8, 5)) # Sets the chart size (8 inches wide, 5 inches tall).
plt.bar(df["Entity"], df["Articles Mentioned In"]) # Creates a bar chart — each entity on the x-axis, number of articles on the y-axis.
plt.title("Trending Entities Based on Number of Articles") # Adds the main title to the chart.
# Label the x and y axes.
plt.xlabel("Entity") 
plt.ylabel("Articles Mentioned In")
plt.xticks(rotation=45, ha="right") # Rotates entity names 45° so they don’t overlap.
# ha = horizontal alignment, telling Matplotlib to align the labels to the right side of their tick mark.
plt.tight_layout() # Adjusts spacing automatically for a clean look.
plt.show() # Displays the final chart window.
