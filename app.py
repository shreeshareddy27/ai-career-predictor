import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer and model
model = load_model('career_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Reverse token index
reverse_map = {v: k for k, v in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index) + 1

# Skills associated with job roles (sample mapping)
skill_map = {
    'data scientist': ['python', 'machine learning', 'pandas'],
    'consultant': ['client communication', 'powerpoint', 'analysis'],
    'software engineer': ['java', 'git', 'algorithms'],
    'data analyst': ['sql', 'excel', 'tableau'],
    'ml engineer': ['tensorflow', 'keras', 'deep learning']
}

# Streamlit UI
st.title("🔮 AI Career Path Predictor")
st.write("Enter your previous job titles (comma-separated):")

user_input = st.text_input("Job History", "intern, data analyst")

if st.button("Predict Next Role"):
    job_titles = [title.strip().lower() for title in user_input.split(',') if title.strip()]
    input_seq = tokenizer.texts_to_sequences([' '.join(job_titles)])
    input_padded = pad_sequences(input_seq, maxlen=model.input_shape[1], padding='post')
    
    predicted_prob = model.predict(input_padded)
    predicted_index = np.argmax(predicted_prob)
    predicted_title = reverse_map.get(predicted_index, "Unknown")
    
    st.success(f"🧠 Predicted next job title: **{predicted_title.title()}**")

    # Recommend skills
    recommended_skills = skill_map.get(predicted_title.lower())
    if recommended_skills:
        st.info("🛠 Recommended Skills:")
        for skill in recommended_skills:
            st.write(f"- {skill}")
