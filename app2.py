import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("final_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🌪️", layout="wide")

# -----------------------------
# Sidebar Section
# -----------------------------
st.sidebar.title("📌 Info & Fun Facts")

# Fun Facts / Stats about Disasters
st.sidebar.markdown("### 🌍 Fun Facts & Stats")
st.sidebar.markdown("""
- On average, over **150 natural disasters** occur worldwide each year.  
- **Floods** and **storms** are the most common disasters.**.  
""")

# Instructions / Info about the App
st.sidebar.markdown("### 📝 How to Use This App")
st.sidebar.markdown("""
1. Enter a tweet in the text box.  
2. Click 🔍 **Predict** to classify it.  
3. 🚨 Red = Disaster | ✅ Green = Not Disaster.  
""")

# -----------------------------
# Main Section
# -----------------------------
st.title("Disaster Tweet Classifier")
st.markdown("Classify tweets as **Disaster** or **Not Disaster**")

# Input text
tweet = st.text_area("Enter tweet here:", placeholder="e.g. Earthquake just hit the city...")

# Predict button
if st.button("🔍 Predict"):
    if tweet.strip():
        # Transform and predict
        X_new = vectorizer.transform([tweet])
        prediction = model.predict(X_new)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This tweet is about a **Disaster**")
        else:
            st.success("✅ This tweet is **Not Disaster related**")
    else:
        st.warning("⚠️ Please enter a tweet before predicting!")
