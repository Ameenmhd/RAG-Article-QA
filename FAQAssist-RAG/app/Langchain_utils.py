from langchain_core.language_models import LLM
from langchain_core.outputs import Generation
from typing import List, Optional, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment.")

genai.configure(api_key=api_key)

class GeminiLLM(LLM):
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

llm = GeminiLLM()

embeddings_hg = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = r"C:\Users\ASUS\Documents\LangChain\3\faiss_index"

def create_vecdb():
    loader = CSVLoader(file_path=r"C:\Users\ASUS\Documents\LangChain\3\codebasics_faqs.csv", source_column="prompt")
    docs = loader.load()
    vecdb = FAISS.from_documents(documents=docs, embedding=embeddings_hg)
    vecdb.save_local(db_path)

def qa_chain():
    vecdb = FAISS.load_local(db_path, embeddings_hg, allow_dangerous_deserialization=True)
    retriever = vecdb.as_retriever()

    prompt_template = """You are a helpful AI assistant. Act professionally and assist users with the information provided in the context.

    If the user asks for help, explain how you can assist them (e.g., by clarifying doubts based on the given information).

    Only use the context below to answer the question. If the answer is not in the context, respond with:
    "I don't know."

    Do not make up facts or provide information beyond the context.

    Context:
    {context}

    Question:
    {question}

    Answer:"""


    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

#if __name__ == "__main__":
#    chain = qa_chain()
#    result = chain.invoke({"query": "do you provide internship? Do you have EMI option?"})
#    print(result)
