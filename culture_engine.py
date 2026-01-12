import os
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class CultureEngine:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Vector Store
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_function,
            collection_name="culture_packs"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def load_culture_pack(self, file_path):
        """Ingests a text file into the vector database."""
        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create Document object
        # We assume the filename implies the culture (e.g., indian_folklore.txt)
        culture_name = os.path.basename(file_path).split(".")[0].replace("_", " ").title()
        
        doc = Document(
            page_content=content,
            metadata={"source": file_path, "culture": culture_name}
        )

        # Split and Add to DB
        chunks = self.text_splitter.split_documents([doc])
        self.vector_store.add_documents(chunks)
        print(f"Successfully loaded {len(chunks)} chunks from {file_path} into ChromaDB.")

    def search(self, query, k=3):
        """Performs semantic search."""
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def get_context_string(self, query):
        """Returns a formatted string of retrieved context."""
        try:
            docs = self.search(query)
            if not docs:
                return ""
            
            context_parts = []
            for doc in docs:
                context_parts.append(f"[Source: {doc.metadata.get('culture', 'Unknown')}]\n{doc.page_content}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Culture Context Retrieval Failed: {e}")
            return ""

# Helper to ingest all packs in data directory
def ingest_all_packs(data_dir="./data/culture_packs"):
    engine = CultureEngine()
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    for f in files:
        engine.load_culture_pack(f)

if __name__ == "__main__":
    # Test Run
    ingest_all_packs()
    ce = CultureEngine()
    print("--- Test Search: 'Festival of Lights' ---")
    print(ce.get_context_string("Festival of Lights"))
