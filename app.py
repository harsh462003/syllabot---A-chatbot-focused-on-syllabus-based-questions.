import streamlit as st  # Importing Streamlit for building the web application
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility  # Importing Milvus library for database operations
from sentence_transformers import SentenceTransformer  # Importing SentenceTransformer for generating embeddings
from PyPDF2 import PdfReader  # Importing PdfReader for reading PDF files
from langchain_community.llms import HuggingFaceHub  # Importing HuggingFaceHub for language model integration
from langchain.prompts import PromptTemplate  # Importing PromptTemplate for creating prompts for the language model
from langchain.schema.runnable import RunnableLambda  # Importing RunnableLambda for creating runnable functions
from docx import Document  # Importing Document for reading DOCX files
import os  # Importing os for environment variable access


# Define connection parameters for Milvus database
MILVUS_HOST = "fill details"  # Host address for Milvus server
MILVUS_PORT = "fill details"  # Port for Milvus server
MILVUS_USER = "fill details"  # Username for Milvus authentication
MILVUS_PASSWORD = "fill details"  # Password for Milvus authentication


def connect_to_milvus():
    """Connect to the Milvus database using the defined parameters."""
    try:
        # Disconnect any existing connection with the alias "default"
        if "default" in connections.list_connections():
            connections.disconnect(alias="default")

        # Connect to the Milvus database
        connections.connect(
            alias="default",
            host=MILVUS_HOST,  # Host address for the Milvus server
            port=MILVUS_PORT,  # Port for the Milvus server
            user=MILVUS_USER,  # Username for Milvus authentication
            password=MILVUS_PASSWORD,  # Password for Milvus authentication
            secure=True  # Use secure connection
        )
        print("✅ Successfully connected to Milvus!")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        raise SystemExit("Terminating script. Please check your credentials or Milvus server status.")


# ❌ Calling connect_to_milvus() before defining MILVUS_* variables
connect_to_milvus()  # This will fail if MILVUS_HOST is not defined before this call


# Define or Load Collection
collection_name = "document_embeddings"  # Name of the collection to store document embeddings


if collection_name not in utility.list_collections():
    # Define fields for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Field for unique ID
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Field for storing embeddings
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=2000),  # Field for storing text chunks
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),  # Field for storing file names
    ]
    schema = CollectionSchema(fields, description="Embedding storage for document chunks")  # Define the schema for the collection
    collection = Collection(name=collection_name, schema=schema)  # Create the collection
    # Create index with COSINE similarity for efficient searching
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
else:
    collection = Collection(name=collection_name)  # Load the existing collection


# Helper Functions
def load_pdf(file_path):
    """Load text from a PDF file and return a list of lowercased text pages."""
    reader = PdfReader(file_path)  # Create a PDF reader object
    return [page.extract_text().strip().lower() for page in reader.pages if page.extract_text()]  # Extract and return text


def load_docx(file_path):
    """Load text from a DOCX file and return a list of lowercased paragraphs."""
    doc = Document(file_path)  # Create a Document object
    return [p.text.strip().lower() for p in doc.paragraphs if p.text.strip()]  # Extract and return text


def chunk_text(text_list, chunk_size=500, overlap=50):
    """Chunk the text into smaller pieces with specified size and overlap."""
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])  # Append each chunk to the list
    return chunks  # Return the list of chunks


def store_embeddings(file_paths):
    """Store embeddings for the provided file paths in the Milvus collection."""
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Load the sentence transformer model
    collection.load()  # Ensure the collection is loaded
    for file in file_paths:
        file_name = file.split("/")[-1]  # Extract the file name from the path
        # Check if embeddings for the file already exist
        if collection.query(f'file_name == "{file_name}"', output_fields=["file_name"]):
            st.info(f"✅ Embeddings for '{file_name}' already exist. Skipping...")
            continue  # Skip to the next file if embeddings exist
        # Load text from the file and chunk it
        chunks = load_pdf(file) if file.endswith(".pdf") else load_docx(file)
        chunks = chunk_text(chunks)  # Chunk the text
        embeddings = model.encode(chunks, convert_to_tensor=False)  # Generate embeddings
        # Match the schema fields for insertion
        data = [embeddings.tolist(), chunks, [file_name] * len(chunks)]
        collection.insert(data)  # Insert data into the collection
        collection.flush()  # Flush the collection to ensure data is saved
        st.success(f"✅ Stored {len(chunks)} chunks from '{file_name}' in Milvus!")  #<diff>



def query_milvus(query, top_k=3, extra_context=3):
    """Query the Milvus collection for relevant information based on the user's query."""
    collection.load()  # Load the collection
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Load the sentence transformer model
    query_embedding = model.encode([query.lower()], convert_to_tensor=False)  # Generate embedding for the query


    # 🛑 Explicit Check for "Leave Request" or "Attendance Shortage"
    if "leave request" in query.lower() or "attendance shortage" in query.lower():
        leave_form_text = "*To apply for leave, fill out the 'Leave Request Form' with your details and obtain faculty approval.*"
        return leave_form_text, ["Leave request form"]  # ✅ Ensure correct source is returned early


    # 🛠 Otherwise, perform a normal search
    search_results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text_chunk", "id", "file_name"]
    )

    if not search_results[0]:
        return "No relevant information found.", None

    # Step 2️⃣: Extract the Top-K Most Relevant Chunks
    matched_chunks = [(res.entity.get("id"), res.entity.get("text_chunk")) for res in search_results[0]]

    # Step 3️⃣: Fetch Extra Surrounding Chunks for Better Context
    surrounding_chunks = []
    for chunk_id, _ in matched_chunks:
        surrounding = collection.query(
            expr=f"id >= {chunk_id - extra_context} and id <= {chunk_id + extra_context}",
            output_fields=["text_chunk", "id", "file_name"]
        )
        surrounding_chunks.extend(surrounding)

    # Step 4️⃣: Merge and Filter Duplicate Chunks
    unique_chunks = {chunk["id"]: chunk["text_chunk"] for chunk in surrounding_chunks}
    sorted_context = "\n\n".join(unique_chunks[key] for key in sorted(unique_chunks))

    # Step 5️⃣: Get all unique file names from the relevant chunks
    file_names = list(set(chunk["file_name"] for chunk in surrounding_chunks if "file_name" in chunk))

    # ✅ Override incorrect sources if the question is about leave request
    if "leave request" in query.lower() or "attendance shortage" in query.lower():
        file_names = ["Leave request form"]  # Force correct source

    # Step 6️⃣: Pass the Enhanced Context to the LLM
    answer = run_llm(query, sorted_context)

    return answer, file_names  # ✅ Now the correct source will be displayed





def run_llm(query, context):
    """Run the language model to generate an answer based on the query and context."""
    llm = HuggingFaceHub(
        repo_id="deepseek-ai/DeepSeek-R1",  # Repository ID for the model
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # API token for Hugging Face Hub
        model_kwargs={"temperature": 0.2, "max_length": 512}  # Model parameters
    )


    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an AI assistant that must ONLY use the provided syllabus context.\n"
                 "However, some of the context may be irrelevant or not directly related to the question.\n\n"
                 "### Step 1️⃣: Filter the Context\n"
                 "- Identify the most relevant part of the syllabus context for answering the question.\n"
                 "- Ignore or remove anything unrelated to the topic of the question.\n\n"
                 "### Step 2️⃣: Generate a Precise Answer\n"
                 "- Answer the question strictly using the relevant syllabus content.\n"
                 "- DO NOT add any extra knowledge.\n"
                 "- DO NOT include references unless they are directly relevant.\n\n"
                 "Question: {question}\n\n"
                 "Syllabus Context:\n{context}\n\n"
                 "Step 1️⃣: Filtered Relevant Context: (AI selects relevant part)\n"
                 "**Step 2️⃣: Final Answer:"
    )

    chain = prompt | llm
    full_response = chain.invoke({"context": context, "question": query})

    # Extract only the final answer
    final_answer = full_response.split("Final Answer: ")[-1].strip()

    return final_answer  # ✅ Ensures only the final answer is returned



pdf_links = {
    "CSE(AI&ML) (Paragraph Syllabus).pdf": "https://drive.google.com/uc?export=download&id=1o-pC6tvFrpy9fo-90HrVtB27QNM5zX-p",
    "Faculty list with email id.pdf": "https://drive.google.com/uc?export=download&id=1orS4Yw79kW34k5hRufREdw5C4BrcaoZ9",
    "Cabin No..pdf": "https://drive.google.com/uc?export=download&id=1uGhL5jpfRl3GEjFt0qAa2evcPc2C_XHq",
    "Leave request form":"https://drive.google.com/uc?export=download&id=1lNOTgDvAxOfke4Q4M8zP4Opwn3DRSKGC",
}

# Streamlit interface for the Academic Chatbot
st.title("📘 Academic Chatbot")  # Set the title of the Streamlit app
st.write("Upload PDF or DOCX files and ask questions!")  # Instructions for the user


uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        store_embeddings([uploaded_file.name])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Create an empty chat history


query = st.text_input("Ask a question:")
file_names = []

if query:
    with st.spinner("Retrieving context..."):
        response, file_names = query_milvus(query)
    st.session_state.chat_history.append({"question": query, "response": response})
    st.markdown(f"### 🤖 Response:\n{response}")

# ✅ Ensure correct source is always set for leave request
if "leave request" in query.lower() or "attendance shortage" in query.lower():
    file_names = ["Leave request form"]  # 🔥 Force correct reference and prevent incorrect ones
elif file_names:
    # ✅ Filter out invalid sources (only display files in pdf_links)
    file_names = [file for file in file_names if file in pdf_links]

# ✅ Display the sources correctly
if file_names:
    st.markdown("### 📄 Sources:")
    for file in file_names:
        st.markdown(f"- [📄 {file}]({pdf_links[file]})", unsafe_allow_html=True)

# Expandable Chat History section
with st.expander("📜 Chat History", expanded=True):  # Create an expandable section for chat history

    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"### 🔹 Q{idx + 1}: {chat['question']}")
            st.markdown(f"Response: {chat['response']}")
            st.markdown("---")
    else:
        st.info("No chat history available yet.")