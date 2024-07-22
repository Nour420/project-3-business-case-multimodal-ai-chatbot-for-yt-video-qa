import os
import streamlit as st
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from uuid import uuid4

# Streamlit layout setup
st.markdown("""
    <style>
        .title {
            font-size: 1.6rem;
        }
        .answer-box {
            border: 2px solid red;
            padding: 10px;
            border-radius: 10px;
            background-color: #262730;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">📄 You can ask me questions related to the context "Health". If you have any questions about immune system, viruses and diseases, cancer, vaccines, or any other topic mentioned in the context, feel free to ask!</h1>', unsafe_allow_html=True)
st.write("Query Projects3 videos Machine Learning. Provide your OpenAI API key and Pinecone API key to start.")

# OpenAI API Key input
openai_api_key = st.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.text_input("Pinecone API Key", type="password")

if not openai_api_key or not pinecone_api_key:
    st.info("Please add your OpenAI and Pinecone API keys to continue.", icon="🗝️")
else:
    # Initialize OpenAI API key
    import openai
    openai.api_key = openai_api_key

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Question input
    question = st.text_area("Ask a question about the transcripts!", placeholder="Can you give me a short summary?")
    if question:
            # Connect to existing Pinecone index
            index_name = 'youtube-videos-data'
            index = pc.Index(index_name)
            
            # Querying
            embed = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key)
            vectorstore = LangchainPinecone(index, embed.embed_query, "transcript")
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.7)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
            
            answer = qa.invoke(question)

            # Extract the result from the answer
            result = answer.get('result', 'No result found.')

            
            # Display the answer in a box
            st.markdown(f"""
                <div class="answer-box">
                    <h4>Answer:</h4>
                    <p>{result}</p>
                </div>
            """, unsafe_allow_html=True)