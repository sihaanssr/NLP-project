import spacy
from pandas import DataFrame


def initialize_entity_model(path="/content/model-best/"):
    nlp = spacy.load(path)
    return nlp


def initialize_relations_model(path="/content/rel_model/model-best"):
    nlp = spacy.load("rel_model/model-best")
    nlp.add_pipe('sentencizer')
    return nlp


def extracts_entities(nlp, text):
    # entity_lists = []
    text=[text]
    for doc in nlp.pipe(text, disable=["tagger"]):
        e_list = [(e.start, e.text, e.label_) for e in doc.ents]
    df = DataFrame(e_list)
    df.columns = ["Position","Text","Label"]
    return df
