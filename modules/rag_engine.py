# some imports needed
# pip install datasets chromadb sentence-transformers wikipedia

import chromadb
import wikipedia
from wikipedia import WikipediaPage # just for type checking
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# hash each entry in the vector db, to get less redundancies
import hashlib

class ContextualizationEngine:
    def __init__(self, chroma_path="./chroma_db", collection_name="knowledge_base"):
        """
        Initialize the RAG engine with ChromaDB and the specific SentenceTransformer model.
        """
        # 1. Initialize the Vector Database (ChromaDB)
        # Using a persistent client saves data to disk. Change path as needed.
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # 2. Load the specific embedding model requested
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"📥 Attempting to load model: {model_name}")
        self.embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # 3. Create or Get the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_func,
            metadata={"hnsw:space": "cosine"} # Cosine similarity is good for text
        )
        print(f"✅ Engine initialized with model: {model_name}")


    def handle_topic_ambiguity(self, topic) -> WikipediaPage:
        """
        Handles ambiguity in the ingestion process.
        """
        try:
            page = wikipedia.page(topic, auto_suggest=True)
            return page
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle Ambiguity
            best_match = e.options[0] # Wikipedia API sorts by relevance, so 0 is usually best
            
            print(f"⚠️ Ambiguous topic '{topic}'.")
            print(f"   ↳ Automatically switching to best match: '{best_match}'")
            print(f"   ↳ (Other options found: {e.options[1:5]})")
            
            # Retry fetching with the specific title
            try:
                page = wikipedia.page(best_match, auto_suggest=False)
                return page
            except Exception as nested_error:
                print(f"❌ Failed to retrieve the auto-selected topic '{best_match}'.")
                raise nested_error

    def ingest_wikipedia_topic(self, topic):
        """
        Fetches a Wikipedia page, chunks the text, and stores it in the Vector DB.
        """
        print(f"📥 Fetching and processing: {topic}...")
        try:
            # Fetch full page
            page = self.handle_topic_ambiguity(topic)

            content = page.content
            url = page.url
            
            # chunking strategy (Simple paragraph split)
            # TODO: In production, use a sliding window or recursive splitter (e.g., LangChain)
            chunks = [c for c in content.split('\n\n') if len(c) > 50] 
            
            ids = [self.generate_hash_id(text=c) for c in chunks]
            metadatas = [{"source": "wikipedia", "url": url, "topic": topic} for _ in chunks]

            # Add to ChromaDB
            self.collection.upsert( # upsert for safe updates
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Successfully indexed {len(chunks)} chunks for '{topic}'.")
        
        except wikipedia.exceptions.PageError:
            print(f"⚠️ Page '{topic}' not found by Wikipedia.")

    def retrieve_context(self, query, n_results=3):
        """
        Searches the database for relevant context based on the query.
        Returns a dictionary containing the Link and a context-fetching function.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Parse results. Chroma returns lists of lists.
        retrieved_items = []
        
        if results['documents']:
            snippets = results['documents'][0]
            for i in range(len(snippets)):
                doc_snippet = snippets[i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                item = {
                    "source_url": metadata.get('url'),
                    "topic": metadata.get('topic'),
                    "context_snippet": doc_snippet, # The requested function
                    "distance": distance
                }
                retrieved_items.append(item)

        return retrieved_items
    
    # Additional Helper Functions
    def generate_hash_id(self, text):
        # Create an MD5 hash of the text content
        # This prevents the same data from being encoded multiple times
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
