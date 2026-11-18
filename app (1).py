import os
import boto3
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from google import genai

# Page Config
st.set_page_config(page_title="Cloud AI Study Assistant", page_icon="☁️")
st.title("☁️ Cloud-Hosted AI Study Assistant")
st.caption("Powered by AWS S3 Object Storage, Streamlit Cloud & Gemini AI")

# Safely load secrets (works both in Colab/local AND on Streamlit Cloud)
def get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
AWS_ACCESS_KEY = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = get_secret("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = get_secret("AWS_BUCKET_NAME")

if not GEMINI_API_KEY:
    st.error("Please configure your API keys in Streamlit Secrets or environment variables.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize AWS S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

def upload_to_s3(file_obj, filename):
    file_obj.seek(0)
    s3_client.upload_fileobj(file_obj, BUCKET_NAME, filename)

def extract_and_chunk(file_obj, chunk_size=500):
    file_obj.seek(0)
    reader = PdfReader(file_obj)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"
    words = full_text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embeddings(texts):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=texts,
    )
    return np.array([e.values for e in response.embeddings], dtype="float32")

# Sidebar for Cloud Upload
with st.sidebar:
    st.header("1. Upload Course Material")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process & Sync to AWS S3"):
        with st.spinner("Uploading to AWS S3 & indexing with Gemini..."):
            # 1. Store file in AWS S3 Cloud Storage
            upload_to_s3(uploaded_file, uploaded_file.name)
            
            # 2. Extract text and create Vector Index
            chunks = extract_and_chunk(uploaded_file)
            embeddings = get_embeddings(chunks)
            index = faiss.IndexFlatL2(embeddings.shape)
            index.add(embeddings)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.success(f"Stored '{uploaded_file.name}' in AWS S3 and indexed {len(chunks)} chunks!")

# Main Q&A Interface
st.header("2. Ask Questions About Your Notes")
question = st.text_input("Enter your question:")

if question:
    if "index" not in st.session_state:
        st.warning("Please upload and process a PDF in the sidebar first.")
    else:
        with st.spinner("Searching cloud index & generating answer..."):
            q_emb = get_embeddings([question])
            _, indices = st.session_state["index"].search(q_emb, k=3)
            context = "\n\n".join([st.session_state["chunks"][i] for i in indices])

            prompt = f"""Answer the question strictly using the provided study notes context.
            If the answer is not in the notes, say so clearly.

            Context:
            {context}

            Question: {question}"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            st.markdown("### Answer")
            st.write(response.text)
