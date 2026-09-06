
import streamlit as st
import torch, re
import pandas as pd
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

st.set_page_config(page_title="MEDGUIDE", page_icon="💊")
st.title("💊 MEDGUIDE")
st.write("AI-based Handwritten Prescription Analyzer")

MODEL_ID = "KushagraWadhwa/medical-prescription-ocr-india"

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return processor, model

@st.cache_data
def load_data():
    ds = load_dataset("dmedhi/indian-medicines", split="train")
    return ds.to_pandas()

processor, model = load_model()
df = load_data()

def clean_name(name):
    name = name.lower()
    name = re.sub(r'\b(tab|tablet|cap|capsule|syp|syrup|inj|injection)\b', '', name)
    name = re.sub(r'\b\d+(\.\d+)?\s*(mg|ml|mcg|g)\b', '', name)
    return ' '.join(re.sub(r'[^a-z\s]', ' ', name).split())

def strength(name):
    name = name.lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(mg|ml|mcg|g)\b', name)
    if m:
        return m.group(1) + m.group(2)
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?=tablet|capsule|tab\b)', name)
    return m.group(1) + "mg" if m else ""

def match_medicine(query):
    q = clean_name(query)
    s = strength(query)

    for _, row in df.iterrows():
        name = clean_name(row["name"])

        if q in name:
            if not s or strength(row["name"]) == s:
                return row

    return None

uploaded = st.file_uploader(
    "Upload prescription image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Prescription", use_container_width=True)

    if st.button("Analyze Prescription"):
        with st.spinner("Reading prescription..."):

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",
                     "text": 'Read ONLY the medicine names and strengths. '
                             'Return ONLY a JSON array like [{"drug":"Telma 40mg"}].'}
                ]
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            raw = processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            medicines = re.findall(
                r'"drug"\s*:\s*"([^"]+)"',
                raw
            )

        st.subheader("Detected Medicines")

        if not medicines:
            st.warning("No medicine names detected.")
        else:
            for med in medicines:
                st.markdown(f"### 💊 {med}")

                row = match_medicine(med)

                if row is None:
                    st.warning("No reliable match found.")
                else:
                    st.success(f"Matched: {row['name']}")
                    st.write("**Composition:**", row["composition"])
                    st.write("**Uses:**", row["uses"])
                    st.write("**Side Effects:**", row["side_effects"])

st.caption(
    "MEDGUIDE is an educational prototype and not a substitute for professional medical advice."
)
