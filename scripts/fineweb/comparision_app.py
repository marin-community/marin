import os
import random
import argparse
import streamlit as st

from datasets import load_dataset
from marin.web.convert import convert_page


# Setting dataset name for comparison
DATASET_NAME='skaramcheti/hello_world_fw'

# Define session configurations
st.set_page_config(layout="wide")


# Load the dataset
@st.cache_resource
def get_dataset():
    progress_bar = st.progress(0)
    dataset = load_dataset(DATASET_NAME, split="train")
    total_rows = len(dataset)
    
    for i in range(total_rows):
        # Update progress bar
        progress_bar.progress((i + 1) / total_rows)
    
    progress_bar.empty()
    return dataset

dataset = get_dataset()
total_lines = len(dataset)


# Main app
def app():
    st.title('Extraction Tools Comparison')
    st.write('This page compares the output of different extraction tools on the same webpage. The tools are: Trafilatura, Readability, and Resiliparse.')
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = random.randint(0, len(dataset) - 1)

    html = dataset[st.session_state.current_index]["text"]
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.subheader("HTML")
        st.text_area("", value=html, height=600, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.subheader("Trafilatura")
        extracted_text = convert_page(html, extract_method="trafilatura")["content"]
        st.text_area("", value=extracted_text, height=600, disabled=True, label_visibility="collapsed")
    
    with col3:
        st.subheader("Readability")
        extracted_text = convert_page(html, extract_method="readability")["content"]
        st.text_area("", value=extracted_text, height=600, disabled=True, label_visibility="collapsed")
    
    with col4:
        st.subheader("Resiliparse")
        extracted_text = convert_page(html, extract_method="resiliparse")["content"]
        st.text_area("", value=extracted_text, height=600, disabled=True, label_visibility="collapsed")

    if st.button("Load Random Comparison"):
        st.session_state.current_index = random.randint(0, total_lines - 1)
        st.rerun()

    st.write(f"Current entry: {st.session_state.current_index + 1}/{total_lines}")

if __name__ == "__main__":
    app()