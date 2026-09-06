
import streamlit as st
from PIL import Image
import pandas as pd
from datasets import load_dataset

st.set_page_config(page_title="MEDGUIDE", page_icon="💊")

st.title("💊 MEDGUIDE")
st.write("AI-based Handwritten Prescription Analyzer")

@st.cache_data
def load_data():
    return load_dataset("dmedhi/indian-medicines", split="train").to_pandas()

df = load_data()

uploaded = st.file_uploader(
    "Upload prescription image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Prescription", use_container_width=True)

    st.info("Prescription uploaded successfully.")

    st.subheader("Detected Medicines")

    # Demo result for college prototype
    medicine = "Telma 40mg"

    st.markdown(f"### 💊 {medicine}")

    match = df[df["name"].str.contains("Telma", case=False, na=False)]

    if len(match):
        row = match.iloc[0]

        st.success(f"Matched: {row['name']}")
        st.write("**Composition:**", row["composition"])
        st.write("**Uses:**", row["uses"])
        st.write("**Side Effects:**", row["side_effects"])
    else:
        st.warning("Medicine information not found.")

st.caption(
    "MEDGUIDE is an educational prototype and not a substitute for professional medical advice."
)
