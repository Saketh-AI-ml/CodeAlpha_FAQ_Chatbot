import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

st.set_page_config(page_title="FAQ Chatbot")

st.title("🤖 FAQ Chatbot")

user_question = st.text_input("Ask your question")

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

if user_question:
    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()

    if similarity[0][best_match] > 0.2:
        st.success("Answer:")
        st.write(answers[best_match])
    else:
        st.warning("Sorry, I couldn't find an answer.")
