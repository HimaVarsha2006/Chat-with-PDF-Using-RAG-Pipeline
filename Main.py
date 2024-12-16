import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of text chunks extracted from the PDF.
    """

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    # Split text into chunks based on paragraphs or other delimiters
    chunks = text.split('\n\n')
    return chunks

def embed_text_chunks(chunks):
    """Embeds text chunks using a pre-trained model.

    Args:
        chunks (list): List of text chunks.

    Returns:
        list: List of embeddings for each chunk.
    """

    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can adjust the model here
    embeddings = model.encode(chunks)
    return embeddings

def create_vector_database(embeddings):
    """Creates a FAISS index for efficient similarity search.

    Args:
        embeddings (list): List of embeddings.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_and_retrieve(query, index):
    """Queries the vector database and retrieves relevant chunks.

    Args:
        query (str): User's natural language query.
        index (faiss.IndexFlatL2): FAISS index.

    Returns:
        list: List of retrieved chunks.
    """

    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can adjust the model here
    query_embedding = model.encode([query], batch_size=1)[0]
    distances, indices = index.search(query_embedding.reshape(1, -1), k=5)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def generate_response(query, retrieved_chunks, model_name="t5-base", max_length=1024):
    """Generates a response using an LLM.

    Args:
        query (str): User's natural language query.
        retrieved_chunks (list): List of retrieved chunks.
        model_name (str, optional): Name of the pre-trained seq2seq model. Defaults to "t5-base".
        max_length (int, optional): Maximum allowed input sequence length. Defaults to 1024.

    Returns:
        str: Generated response.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Truncate retrieved chunks if necessary
    truncated_chunks = [chunk[:max_length] for chunk in retrieved_chunks]
    joined_text = " ".join(truncated_chunks)

    # Improved prompt engineering:
    if "summarize" in query.lower():
        prompt = f"Summarize the following text concisely:\n\n{joined_text}"
    elif "?" in query:
        prompt = f"Answer the following question based on the provided context:\n\nQuestion: {query}\n\nContext: {joined_text}"
    else:
        prompt = f"Provide a comprehensive response to the following query, leveraging the information in the provided context:\n\nQuery: {query}\n\nContext: {joined_text}"

    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while processing your query. Please try again later."

    return generated_text

if __name__ == "__main__":
    pdf_path = "Hello.pdf"  # Replace with your PDF file path
    chunks = extract_text_from_pdf(pdf_path)
    embeddings = embed_text_chunks(chunks)
    index = create_vector_database(embeddings)

    query = input("Enter your query: ")
    retrieved_chunks = query_and_retrieve(query, index)
    response = generate_response(query, retrieved_chunks)

    print(response)