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
    input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(input_tokenized, num_beams=4, length_penalty=0.1, min_length=64, max_length=256)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    return summary

# Streamlit code
st.set_page_config(page_title="Legal AI Summarizer", page_icon="img.png")
title = "Legal AI Summarizer"
st.title(title)
st.write("This is a legal AI summarizer. Simply copy and paste your legal document below to get a humanized, concise summary.")

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

user_text = st.text_area("Paste your legal document here:", value=st.session_state.user_text, height=300)

if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        try:
            summary_text = summarize(led_in, led_in_tokenizer, user_text)
            st.session_state.user_text = user_text
            st.write("")
            st.success(summary_text)
        except Exception as e:
            st.error(f"An error occurred: {e}")