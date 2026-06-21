import streamlit as st
import tempfile, os, time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Q&A Assistant")
st.caption(
    "Retrieval-Augmented Generation (RAG) powered document assistant"
)

with st.sidebar:
    st.header("⚙️ Setup")
    st.success("✅ Ready to use")

    st.markdown("---")

    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        **PDF Q&A Assistant**

        Upload a PDF and ask questions.

        Powered by:
        - Llama 3.1
        - FAISS
        - MiniLM Embeddings
        - LangChain
        - Streamlit
        """)

    st.markdown("---")

    st.info(
        "All processing happens locally except the final LLM call to Groq."
    )


@st.cache_resource  
def build_vector_store(pdf_bytes: bytes, file_name: str):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    full_text = "\n".join(
        [doc.page_content for doc in documents]
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    os.unlink(tmp_path)
    
    return vector_store, len(chunks), full_text, documents

def build_qa_chain(vector_store, api_key: str):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  
    )
    
    prompt_template = """
You are a helpful PDF assistant.

Use ONLY the information provided in the document context to answer the user's question.

Instructions:
- Give clear and detailed answers when the information is available.
- Summarize information when appropriate.
- If multiple relevant points exist, present them in bullet points.
- Be specific and use details from the document.
- Do not invent facts that are not present in the context.
- If the answer is genuinely unavailable in the context, respond with:
  "I couldn't find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1  
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",      
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True 
    )
    
    return qa_chain

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="Upload any PDF — textbook, notes, research paper, resume, etc."
)

if uploaded_file is not None:
    
    with st.spinner("🔄 Reading PDF and building vector index... (first time takes ~30 seconds)"):
        vector_store, num_chunks, full_text, documents = build_vector_store(
            uploaded_file.read(),
            uploaded_file.name    
        )
        st.session_state.documents = documents
        st.success(
            f"✅ PDF processed! Created {num_chunks} searchable chunks."
        )

    qa_chain = build_qa_chain(
        vector_store,
        st.secrets["GROQ_API_KEY"]
    )
    
    if st.button("📋 Generate PDF Summary"):

        summary_prompt = """
        Provide a detailed summary of the entire document.

        Include:
        1. Main topic
        2. Key concepts
        3. Important points
        4. Conclusion
        """

        with st.spinner("Generating summary..."):
            summary_result = qa_chain.invoke({
                "query": summary_prompt
            })

        st.subheader("📋 Document Summary")
        st.write(summary_result["result"])

if "messages" not in st.session_state:
    st.session_state.messages = []
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("🗑 Clear"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask anything about the PDF...")

if uploaded_file is not None and user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document and generating answer..."):
            start_time = time.time()
            
            import re

            page_match = re.search(
                r"(?:page|pg)\s*(\d+)",
                user_question.lower()
            )

            if page_match:
                page_num = int(page_match.group(1))

                documents = st.session_state.documents

                if 1 <= page_num <= len(documents):

                   page_text = documents[page_num - 1].page_content

                   st.write(f"### 📄 Page {page_num}")
                   st.write(page_text)

                else:
                   st.error(
                       f"Page {page_num} does not exist. This PDF has only {len(documents)} pages."
                   )

                st.stop()


            result = qa_chain.invoke({"query": user_question})
            

            end_time = time.time()

            answer = result["result"]
            source_docs = result["source_documents"]

            response_time = round(end_time - start_time, 2)

            pages = []

            for doc in source_docs:
                if "page" in doc.metadata:
                    pages.append(doc.metadata["page"] + 1)

            pages = sorted(list(set(pages)))

            st.write(answer)

            if pages:
                st.caption(
                    f"📄 Source Pages: {', '.join(map(str, pages))}"
                )

            st.caption(
                f"⏱️ Response Time: {response_time} sec"
            )

            with st.expander(
                "📑 Sources used (retrieved chunks)",
                expanded=False
            ):
                for i, doc in enumerate(source_docs, 1):
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Chunk {i}** (Page {page})")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

else:
    st.info(
        "👆 Upload a PDF to get started. Try with any notes, textbook, or document!"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "LLaMA 3.1")

    with col2:
        st.metric("Embeddings", "MiniLM-L6")

    with col3:
        st.metric("Vector DB", "FAISS")