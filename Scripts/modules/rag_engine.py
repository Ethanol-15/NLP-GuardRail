# some imports needed
# pip install datasets chromadb sentence-transformers wikipedia

import chromadb
import wikipedia
from wikipedia import WikipediaPage # just for type checking
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# hash each entry in the vector db, to get less redundancies
import hashlib

from enum import Enum

class ContextualizationEngine:
    def __init__(self, chroma_path="./chroma_db", collection_name="knowledge_base"):
        """
        Initialize the RAG engine with ChromaDB and the specific SentenceTransformer model.
        """
        # 1. Initialize the Vector Database (ChromaDB)
        # Using a persistent client saves data to disk. Change path as needed.
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
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

    def ingest_wikipedia_topic(self, topic, max_chunks=3):
        """
        Fetches a Wikipedia page, chunks the text, and stores it in the Vector DB. 
        
        **NOTE:** Topic should be as direct as possible.
        Params:
            topic: The topic to fetch and ingest from Wikipedia (e.g. Faker (gamer), Kim Jong Un)
            max_chunks: The maximum number of chunks to ingest (default: 3)
        """
        print(f"📥 Fetching and processing: {topic}...")
        try:
            # Fetch full page
            page = self.handle_topic_ambiguity(topic)

            content = page.content
            url = page.url
            
            # chunking strategy (Simple paragraph split)
            # TODO: In production, use a sliding window or recursive splitter (e.g., LangChain)
            all_chunks = [c for c in content.split('\n\n') if len(c) > 50]

            # Wikipedia has an inverted pyramid structure 
            # this means that most of the important information is at the top
            top_chunks = all_chunks[:max_chunks]
            
            ids = [self.generate_hash_id(text=c) for c in top_chunks]
            metadatas = [{"source": "wikipedia", "url": url, "topic": topic} for _ in top_chunks]

            # Add to ChromaDB
            self.collection.upsert( # upsert for safe updates
                documents=top_chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Successfully indexed {len(top_chunks)} chunks for '{topic}'.")
        
        except wikipedia.exceptions.PageError:
            print(f"⚠️ Page '{topic}' not found by Wikipedia.")

    def retrieve_context(self, query, n_results=3, ingest_if_needed=True):
        """
        Searches the database for relevant context based on the query.
        Returns a dictionary containing the Link and a context-fetching function.

        **NOTE:** Query should be as direct as possible.
        Params:
            query: The query to search for
            n_results: The number of results to return
            ingest_if_needed: Whether to ingest the topic if it's not found in the DB
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # if there are no good results then we need to ingest the query
        if ingest_if_needed == True:
            if results is not None and results['documents']:
                distance_len = len(results['distances'][0])
                total_rating = 0
                for i in range(distance_len):
                    distance = results['distances'][0][i]
                    rating = ContextualizationEngine.interpret_distance(distance)
                    total_rating += rating.value
                
                # Re-ingest the query if the results are not relevant enough
                if self.has_strong_enough_relevance(total_rating) == False:
                    print("Not enough relevant documents found, re-ingesting query...")
                    self.ingest_wikipedia_topic(query)
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=n_results
                    )

        # Parse results. Chroma returns lists of lists.
        retrieved_items = []
        
        if results is not None and results['documents']:
            result_len = len(results['documents'][0])
            for i in range(result_len):
                doc_snippet = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                def get_snippet(text=doc_snippet):
                    return text

                item = {
                    "source_url": metadata.get('url'),
                    "topic": metadata.get('topic'),
                    "get_context_snippet": get_snippet, # The requested function
                    "distance": distance
                }
                retrieved_items.append(item)

        return retrieved_items
    
    # Additional Modules
    def generate_hash_id(self, text):
        """
        Create an MD5 hash of the text content. 
        This prevents the same data from being encoded multiple times
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def handle_topic_ambiguity(self, topic) -> WikipediaPage:
        """
        Handles ambiguity in the ingestion process.
        Params:
            topic: The topic to fetch or disambiguate if needed
        Returns:
            A wikipedia page object
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

    # Add specific distance ratings

    class DISTANCE_RATING(Enum):
        """
        Enum for the distance rating categories. 
        This also assigns a specific strength to each category.
        """
        STRONG = 8
        GOOD = 5
        WEAK = 1
        IRRELEVANT = 0

    @staticmethod
    def has_strong_enough_relevance(distance_rating) -> bool:
        """
        Checks if the total distance rating is strong enough to be considered relevant.
        Returns a boolean.
        Params:
            distance_rating: The total distance rating of the given documents.
        """
        # this is easier to pass the more documents are retrieved.
        return distance_rating >= ContextualizationEngine.DISTANCE_RATING.STRONG.value
    
    @staticmethod
    def interpret_distance(distance) -> DISTANCE_RATING:
        """
        Interprets the cosine similarity distance between 0 and 1 as a rating.
        Returns a ContextualizationEngine.DISTANCE_RATING Enum.
        Params:
            distance: The cosine similarity distance between 0 and 1
        """

        if distance < 0.20:
            return ContextualizationEngine.DISTANCE_RATING.STRONG
        elif distance < 0.45:
            return ContextualizationEngine.DISTANCE_RATING.GOOD
        elif distance < 0.60:
            return ContextualizationEngine.DISTANCE_RATING.WEAK
        else:
            return ContextualizationEngine.DISTANCE_RATING.IRRELEVANT
        