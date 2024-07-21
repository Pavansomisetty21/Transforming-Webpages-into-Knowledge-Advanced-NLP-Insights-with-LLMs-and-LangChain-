import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
import os

# Set the API key directly in the code
api_key = "Your api key"  # Replace with your actual API key

# Set the USER_AGENT environment variable
os.environ["USER_AGENT"] = "YourAppName/1.0 (contact@example.com)"

# Initialize LangChain model with Gemini API
llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")

# Define Streamlit UI
st.title("Web Page Analysis")

# URL input
url = st.text_input("Enter Web Page URL:")

if url:
    # Load the web page
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Function to summarize the content
    def summarize_content(docs):
        template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke({"input_documents": docs})
        return response["output_text"]
    
    # Function to generate QA pairs
    def generate_qa_pairs(docs):
        template = """Generate question-answer pairs from the following text:
        "{text}"
        QUESTION:
        ANSWER:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke({"input_documents": docs})
        return response["output_text"]

    # Function to perform data extraction
    def extract_data(docs):
        template = """Extract all text data from the following web page:
        "{text}"
        EXTRACTED DATA:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke({"input_documents": docs})
        return response["output_text"]
    
    # Function to answer questions
    def answer_question(docs, question):
        template = """Generate an answer to the following question based on the text:
        "{text}"
        QUESTION: {question}
        ANSWER:"""
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        input_data = {
            "input_documents": docs,
            "input_question": question,
        }
        response = stuff_chain.invoke(input_data)
        return response["output_text"]

    # Option for user to select functionality
    option = st.selectbox("Select functionality:", ["Summarize", "Generate QA Pairs", "Extract Data", "Answer Question"])

    if option == "Summarize":
        if st.button("Generate Summary"):
            summary = summarize_content(docs)
            st.write("**Summary:**")
            st.write(summary)

    elif option == "Generate QA Pairs":
        if st.button("Generate QA Pairs"):
            qa_pairs = generate_qa_pairs(docs)
            st.write("**QA Pairs:**")
            st.write(qa_pairs)

    elif option == "Extract Data":
        if st.button("Extract Data"):
            extracted_data = extract_data(docs)
            st.write("**Extracted Data:**")
            st.write(extracted_data)

    elif option == "Answer Question":
        question = st.text_input("Enter your question:")
        if st.button("Generate Answer"):
            if question:
                answer = answer_question(docs, question)
                st.write("**Answer:**")
                st.write(answer)
            else:
                st.error("Please enter a question.")
