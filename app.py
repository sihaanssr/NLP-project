from re import sub
import streamlit as st
from defaults import DEFAULT_TEXT
from nlp_scripts import initialize_entity_model, initialize_relations_model, extracts_entities

# status_text = st.empty()
# status_text.text()
# status_text.text("Loading NER model done")

with st.spinner("Loading Entity Recognition Model..."):
    nlp = initialize_entity_model()
st.success("finished loading NER module")

# with st.spinner("Loading Entity Recognition Model..."):
#     relations = initialize_relations_model()
# st.success("finished loading NER module")


st.title("Joint Entities and Relation Extractor")

with st.form(key='my_form'):
    text_input = st.text_area(
        'Enter the job description to extract entities', DEFAULT_TEXT)
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    entities = extracts_entities
    st.dataframe(entities)
