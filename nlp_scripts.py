import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from scripts.rel_pipe import make_relation_extractor, score_relations
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
from pandas import DataFrame
# import streamlit as st


def initialize_entity_model(path="/content/model-best/"):
    nlp = spacy.load(path)
    return nlp


def initialize_relations_model(path="/content/rel_model/model-best"):
    nlp = spacy.load(path)
    nlp.add_pipe('sentencizer')
    return nlp


def extracts_entities(nlp, text):
    # entity_lists = []
    text = [text]
    for doc in nlp.pipe(text, disable=["tagger"]):
        e_list = [(e.start, e.text, e.label_) for e in doc.ents]
    df = DataFrame(e_list)
    df.columns = ["Position", "Text", "Label"]
    return df, doc


def classify_relations(relation_model, doc, chosen_relation, threshold):
    for _, proc in relation_model.pipeline:
        doc = proc(doc)

    relations = []
    for value, rel_dict in doc._.rel.items():
        for sent in doc.sents:
            for e in sent.ents:
                for b in sent.ents:
                    if e.start == value[0] and b.start == value[1]:
                        if rel_dict[chosen_relation] >= threshold:
                            relations.append(
                                [e.text, b.text, *rel_dict.values()])
    df = DataFrame(relations, columns=[
                   "First Entity", "Second Entity", "Conf. DEGREE_IN", "Conf. EXPERIENCE_IN"])
    return df
