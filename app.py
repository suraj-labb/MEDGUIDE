
import streamlit as st
import torch, re, json
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
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return processor, model

@st.cache_data
def load_data():
    return load_dataset("dmedhi/indian-medicines", split="train").to_pandas()

processor, model = load_model()
df = load_data()

uploaded = st.file_uploader(
    "Upload prescription image", type=["jpg", "jpeg", "png"]
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
                    {"type": "text", "text":
                     'Read all medicine names and strengths. '
                     'Return ONLY JSON: [{"drug":"medicine name"}]'}
                ]
            }]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False
                )

            raw = processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            match = re.search(r'\[.*\]', raw, re.S)

            try:
                medicines = json.loads(match.group()) if match else []
            except:
                medicines = []

        st.subheader("💊 Detected Medicines")

        for item in medicines:
            med = item.get("drug", "").strip()

            if not med:
                continue

            st.markdown(f"### {med}")

            results = df[
                df["name"].str.contains(
                    med.split()[0], case=False, na=False
                )
            ]

            if results.empty:
                st.warning("No reliable medicine match found.")
            else:
                row = results.iloc[0]
                st.success(f"Matched: {row['name']}")
                st.write("**Composition:**", row["composition"])
                st.write("**Uses:**", row["uses"])
                st.write("**Side Effects:**", row["side_effects"])

st.caption(
    "MEDGUIDE is an educational prototype and not a substitute for professional medical advice."
)
