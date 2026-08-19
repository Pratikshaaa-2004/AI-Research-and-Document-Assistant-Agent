import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------- STEP 1: Load the PDF ----------
pdf_path = "data/04-CMS-CRM-Analytics-and-Automation.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    print("Please make sure the PDF is directly inside the 'data' folder (not inside .venv).")
else:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Step 1 done: Loaded {len(docs)} pages from the PDF")

    # ---------- STEP 2: Split into chunks ----------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Step 2 done: Split into {len(chunks)} chunks")

    # ---------- STEP 3: Create embeddings (free, runs locally) ----------
    print("Step 3: Creating embeddings... (this may take a minute the first time)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ---------- STEP 4: Store in Chroma vector database ----------
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Step 4 done: Chunks embedded and stored in ChromaDB")

    # ---------- Quick test: search the database ----------
    query = "What is CRM?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"\nTest search for: '{query}'")
    print("Top result preview:")
    print(results[0].page_content[:300])

    print("\n✅ Part A complete — your document is now searchable!")