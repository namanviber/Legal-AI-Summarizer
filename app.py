import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Loading LED IN Model
base_model = "nsi319/legal-led-base-16384"
led = AutoModelForSeq2SeqLM.from_pretrained(base_model)
adapter_model_in = f"Legal-LED_IN_ABS"
led_in = PeftModel.from_pretrained(led, adapter_model_in)
led_in_tokenizer = AutoTokenizer.from_pretrained(base_model)

# Generating Summary
def summarize(model, tokenizer, text):
    input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=8192, truncation=True)
    summary_ids = model.generate(input_tokenized, num_beams=4, length_penalty=0.1, min_length=32, max_length=512)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    return summary

# Reading Txt File
def read_txt_file(file):
    text = file.read().decode('utf-8')
    return text

st.set_page_config(page_title="Legal AI Summarizer", page_icon="img.png")
title = "Legal AI Summarizer"
col1, col2 = st.columns([1,7])
with col1:
    st.image("img.png")
with col2: st.title(title)
st.write("Stuck with long legal documents? Our AI summarizer can help!  Just copy-paste the text or upload a .txt file, and it will give you a quick and easy summary in plain English, so you can understand the key points without all the legalese.")

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

upload_file = st.file_uploader("Upload a .txt file", type="txt")

if upload_file is not None:
    user_text = read_txt_file(upload_file)
else:
    user_text = st.text_area("Paste your legal document here:", value=st.session_state.user_text, height=300)

if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        try:
            summary_text = summarize(led_in, led_in_tokenizer, user_text)
            st.session_state.user_text = user_text
            st.write("")
            st.success(summary_text)
            print(summary_text)
        except Exception as e:
            st.error(f"An error occurred: {e}")