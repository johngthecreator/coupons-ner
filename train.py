import random
import polars as pl
from spacy.tokens import DocBin
import spacy


def create_spacy_full_data():
    # Load your CSV file
    df = pl.read_csv("./data/brands_0408.csv")

    # Initialize a blank spaCy model
    nlp = spacy.blank("en")
    doc_bin = DocBin()  # This will hold the training data


    for product_name, brands in df.iter_rows():
        text = product_name
        brands = brands.split(", ")  # Split multiple brands by comma
        doc = nlp.make_doc(text)
        
        # Create entity annotations
        entities = []
        for brand in brands:
            start = text.casefold().find(brand.casefold())
            # if "Mr. Clean" in product_name:
            #     print(text.casefold())
            #     print(brand.casefold())
            #     print(start)
            #     print(start + len(brand))
            if start != -1:  # Ensure the brand exists in the sentence
                end = start + len(brand)
                entities.append((start, end, "BRAND"))  # Label all brands as "BRAND"

            
        
        # Set entities on the doc
        spans = [doc.char_span(start, end, label) for start, end, label in entities]
        doc.ents = [span for span in spans if span is not None]  # Filter out None spans

        # Add the doc to the DocBin
        doc_bin.add(doc)

    doc_bin.to_disk("./full_data.spacy")
    print("data created")

def split_data():
    # Load the full data
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk("./full_data.spacy")
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Shuffle and split the data
    random.shuffle(docs)
    split = int(len(docs) * 0.8)  # 80% training, 20% validation
    train_docs = docs[:split]
    dev_docs = docs[split:]

    # Save the splits as .spacy files
    train_doc_bin = DocBin(docs=train_docs)
    train_doc_bin.to_disk("./train.spacy")

    dev_doc_bin = DocBin(docs=dev_docs)
    dev_doc_bin.to_disk("./dev.spacy")

def show_spacy_train():
    # Load your spaCy pipeline (or a blank model)
    nlp = spacy.blank("en")

    # Load the .spacy file
    doc_bin = DocBin().from_disk("./full_data.spacy")

    # Pass nlp.vocab (not nlp) to get_docs
    docs = doc_bin.get_docs(nlp.vocab)

    # Iterate through the docs to inspect them
    for doc in docs:
        if len(doc.ents) == 0:
            print("Text:", doc.text)

create_spacy_full_data()