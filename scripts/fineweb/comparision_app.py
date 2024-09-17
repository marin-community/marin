import json
import random
import streamlit as st

st.set_page_config(layout="wide")

@st.cache_resource
def get_file_positions():
    positions = []
    with open('hello_world_fw_extracted.jsonl', 'rb') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            positions.append(pos)
    return positions

def load_single_entry(index):
    positions = get_file_positions()
    with open('hello_world_fw_extracted.jsonl', 'rb') as f:
        f.seek(positions[index])
        line = f.readline().decode('utf-8')
        return json.loads(line)

file_positions = get_file_positions()
total_lines = len(file_positions)

def app():
    st.title('Extraction Tools Comparison')
    st.write('This page compares the output of different extraction tools on the same webpage. The tools are: Trafilatura, Readability, and Resiliparse.')
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = random.randint(0, total_lines - 1)

    data = load_single_entry(st.session_state.current_index)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.subheader("HTML")
        st.text_area("", value=data['html'], height=600, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.subheader("Trafilatura")
        st.text_area("", value=data['trafilatura'], height=600, disabled=True, label_visibility="collapsed")
    
    with col3:
        st.subheader("Readability")
        st.text_area("", value=data['readability'], height=600, disabled=True, label_visibility="collapsed")
    
    with col4:
        st.subheader("Resiliparse")
        st.text_area("", value=data['resiliparse'], height=600, disabled=True, label_visibility="collapsed")

    if st.button("Load Random Comparison"):
        st.session_state.current_index = random.randint(0, total_lines - 1)
        st.rerun()

    st.write(f"Current entry: {st.session_state.current_index + 1}/{total_lines}")

if __name__ == "__main__":
    app()