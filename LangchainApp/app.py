import streamlit as st
import os
import asyncio
import sys
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_mistralai.embeddings import MistralAIEmbeddings

# Set up proper event loop for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# Get environment variables
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def load_mistral_model():
    if not HF_API_KEY:
        st.error("⚠️ Hugging Face API Key is missing! Set it in your .env file as HUGGINGFACEHUB_API_TOKEN")
        return None

    try:
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            huggingfacehub_api_token=HF_API_KEY,
            temperature=0.7,
            max_new_tokens=500,  # Changed from max_length to max_new_tokens
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )
    except Exception as e:
        st.error(f"🚨 Error loading Mistral model: {str(e)}")
        return None


# Sidebar
with st.sidebar:
    st.title('📄 Chat with PDF (LLM)')
    st.markdown("""
    ## About
    This app allows you to chat with a PDF document using Mistral-7B.
    """)
    add_vertical_space(3)
    st.write("👨‍💻 **Developed by:** Abhay and Lekhan")

    # Add model parameters
    temperature = st.slider("Response creativity", 0.1, 1.0, 0.7)
    max_length = st.slider("Max response length", 100, 1000, 500)


def main():
    st.header('Chat with PDF')

    pdf = st.file_uploader("Upload your PDF file", type='pdf')

    if pdf is not None:
        # Process PDF
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text

            if not text:
                st.error("No text could be extracted from the PDF.")
                return

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Initialize embeddings
            try:
                embeddings = MistralAIEmbeddings(
                    mistral_api_key=MISTRAL_API_KEY,
                    model="mistral-embed"
                )
            except Exception as e:
                st.error(f"Failed to initialize embeddings: {str(e)}")
                return

            # Vector store
            store_name = pdf.name[:-4]
            faiss_path = f"vectorstores/{store_name}_faiss"
            os.makedirs("vectorstores", exist_ok=True)

            if os.path.exists(faiss_path):
                try:
                    vectorStore = FAISS.load_local(
                        faiss_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    st.error(f"Failed to load vector store: {str(e)}")
                    return
            else:
                try:
                    vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    vectorStore.save_local(faiss_path)
                except Exception as e:
                    st.error(f"Failed to create vector store: {str(e)}")
                    return

        # Chat interface
        query = st.text_input("Ask your question about the PDF:")

        if query:
            # Initialize model here to use sidebar parameters
            mistral_llm = load_mistral_model()
            if not mistral_llm:
                return

            with st.spinner("Searching for answers..."):
                try:
                    docs = vectorStore.similarity_search(query=query, k=5)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    prompt = f"""
                    [INST] You are a helpful AI assistant that answers questions based on the provided context.

                    Context:
                    {context}

                    Question: {query}

                    Please provide a detailed answer based on the context. If the answer isn't in the context, say "I don't know".

                    Answer: [/INST]
                    """

                    response = mistral_llm(prompt)  # Changed from invoke() to direct call

                    # st.subheader("🔎 Relevant Document Sections:")
                    # for i, doc in enumerate(docs, 1):
                    #     with st.expander(f"Section {i}"):
                    #         st.write(doc.page_content)

                    st.subheader("🤖 Mistral-7B Response:")
                    response_placeholder = st.empty()

                    # Simulated AI response typing effect
                    display_text = ""
                    for word in response.split():
                        display_text += word + " "
                        response_placeholder.markdown(
                            f"<div style='text-align: justify; font-size: 16px;'>{display_text}</div>",
                            unsafe_allow_html=True)
                        time.sleep(0.05)

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


if __name__ == '__main__':
    main()