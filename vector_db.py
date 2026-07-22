# only for testing

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# # Load PDF
# loader = PyPDFLoader("data/sample.pdf")
# documents = loader.load()

# print("Documents loaded")

# # Split text
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# texts = text_splitter.split_documents(documents)

# print("Text split into chunks")

# # Use smaller stable model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
# )

# # Create vector database
# db = FAISS.from_documents(texts, embeddings)

# # Save database
# db.save_local("vectorstore")

# print("Vector database created successfully")