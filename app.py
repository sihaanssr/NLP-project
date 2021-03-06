# import spacy
# from emoji import demojize
from re import sub
import streamlit as st
from defaults import DEFAULT_TEXT
from nlp_scripts import initialize_entity_model, initialize_relations_model, extracts_entities, classify_relations

# status_text = st.empty()
# status_text.text()
# status_text.text("Loading NER model done")

# nlp = spacy.load('en_core_web_sm')

# def emoji_to_text(line):
#     line = demojize(line)
#     return line


# def preprocess(text_input):
#     # sentence = emoji_to_text(text_input)
#     sentence = nlp(text_input)
    
#     sentence = [word for word in sentence if not word.is_punct]
#     sentence = [word for word in sentence if len(word)>3]
#     sentence = [word for word in sentence if not word.is_stop]
#     sentence = [word for word in sentence if not word.like_url]
#     sentence = [word.lemma_ for word in sentence]
    
#     return " ".join(sentence)


with st.spinner("Loading Entity Recognition Model..."):
    nlp = initialize_entity_model()
st.success("finished loading NER module")

with st.spinner("Loading Relation Categorization Model..."):
    relations = initialize_relations_model()
st.success("finished loading REL module")


st.title("Joint Entities and Relation Extractor")

with st.form(key='my_form'):
    text_input = st.text_area(
        'Enter the job description to extract entities', DEFAULT_TEXT)
    submit_button = st.form_submit_button(label='Submit')

# text_input = preprocess(text_input)

entities, doc = extracts_entities(nlp, text_input)
if submit_button:
    st.dataframe(entities)


with st.form(key='new_form'):
    threshold = st.slider(label="Choose a threshold for the model",
                          min_value=0.5, max_value=1.0, step=0.01)
    chosen_relation = st.selectbox("Choose the relation you want to explore", [
        "EXPERIENCE_IN", "DEGREE_IN"])
    rel_submit_button = st.form_submit_button(label='Explore the relation')

if rel_submit_button:
    df = classify_relations(relations, doc, chosen_relation, threshold)
    st.dataframe(df)
