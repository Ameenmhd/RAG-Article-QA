import streamlit as st
from Langchain import create_vecdb, qa_chain

st.set_page_config(page_title="CodeQA", layout="centered")

st.title("💬 CodeBasicsQA Assistant")


# Button to trigger vector DB creation (optional)
if st.button("Create Vector DB"):
    create_vecdb()
    st.success("Vector database created successfully!")

# Input box for user query
que = st.text_input("Enter your question:")

# When user submits a question
if que:
    chain = qa_chain()
    res = chain({"query": que})  # ✅ Corrected input format

    # Display the result
    st.header("📌 Answer")
    st.write(res["result"])

    
