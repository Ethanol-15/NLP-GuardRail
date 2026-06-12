# some imports needed
# pip install datasets chromadb sentence-transformers wikipedia requests

import hashlib
import requests
from enum import Enum

import chromadb
import wikipedia
from wikipedia import WikipediaPage  # just for type checking
from chromadb.utils import embedding_functions


class ContextualizationEngine:
    def __init__(self, chroma_path="./chroma_db", collection_name="knowledge_base"):
        """
        Initialize the RAG engine with ChromaDB and the specific SentenceTransformer model.
        """

        # 1. Initialize the Vector Database (ChromaDB)
        # Using a persistent client saves data to disk.
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # 2. Load the embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"📥 Attempting to load model: {model_name}")

        self.embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # 3. Create or get the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_func,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"✅ Engine initialized with model: {model_name}")

    def ingest_wikipedia_topic(self, topic, max_chunks=3, lang="en"):
        """
        Fetches a Wikipedia page, chunks the text, and stores it in the Vector DB.

        If Wikipedia fails or returns invalid JSON, the app will not crash.
        Instead, it skips RAG ingestion safely.
        """

        print(f"📥 Fetching and processing: {topic}...")
        wikipedia.set_lang(lang)

        try:
            # Fetch full page
            page = self.handle_topic_ambiguity(topic)

            if page is None:
                print(f"⚠️ No Wikipedia page retrieved for '{topic}'. Skipping ingestion.")
                return False

            content = page.content
            url = page.url

            # Simple paragraph chunking
            all_chunks = [c for c in content.split("\n\n") if len(c) > 50]
            top_chunks = all_chunks[:max_chunks]

            if not top_chunks:
                print(f"⚠️ No useful chunks found for '{topic}'.")
                return False

            ids = [self.generate_hash_id(text=c) for c in top_chunks]

            metadatas = [
                {
                    "source": "wikipedia",
                    "url": url,
                    "topic": topic
                }
                for _ in top_chunks
            ]

            # Add to ChromaDB
            self.collection.upsert(
                documents=top_chunks,
                metadatas=metadatas,
                ids=ids
            )

            print(f"✅ Successfully indexed {len(top_chunks)} chunks for '{topic}'.")
            return True

        except wikipedia.exceptions.PageError:
            print(f"⚠️ Page '{topic}' not found by Wikipedia.")
            return False

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"⚠️ Ambiguous Wikipedia topic '{topic}'. Options: {e.options[:5]}")
            return False

        except requests.exceptions.JSONDecodeError as e:
            print(f"⚠️ Wikipedia returned invalid JSON for '{topic}'. Skipping RAG ingestion.")
            print(f"Error: {e}")
            return False

        except Exception as e:
            print(f"⚠️ Unexpected Wikipedia/RAG ingestion error for '{topic}'.")
            print(f"Error: {e}")
            return False

    def mass_ingest(self, topic_list):
        """
        Mass populate the database with the given list of topics.
        """

        for topic in topic_list:
            self.ingest_wikipedia_topic(topic)

    def retrieve_context(self, query, n_results=3, ingest_if_needed=True):
        """
        Searches the database for relevant context based on the query.

        Returns a list of dictionaries containing:
        - source_url
        - topic
        - get_context_snippet function
        - distance

        If Wikipedia or ChromaDB fails, this function returns a safe fallback context
        instead of crashing the app.
        """

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            # If there are no good results, try to ingest the query from Wikipedia.
            if ingest_if_needed:
                if results is not None and results.get("documents"):
                    documents = results["documents"]

                    if documents and len(documents[0]) > 0:
                        distance_len = len(results["distances"][0])
                        total_rating = 0

                        for i in range(distance_len):
                            distance = results["distances"][0][i]
                            rating = ContextualizationEngine.interpret_distance(distance)
                            total_rating += rating.value

                        # Re-ingest the query if results are not relevant enough
                        if not self.has_strong_enough_relevance(total_rating):
                            print("⚠️ Not enough relevant documents found. Trying Wikipedia ingestion...")

                            success = self.ingest_wikipedia_topic(query)

                            if success:
                                results = self.collection.query(
                                    query_texts=[query],
                                    n_results=n_results
                                )
                            else:
                                print("⚠️ Wikipedia ingestion failed or skipped. Continuing without new RAG context.")

                    else:
                        print("⚠️ No documents found in ChromaDB. Trying Wikipedia ingestion...")

                        success = self.ingest_wikipedia_topic(query)

                        if success:
                            results = self.collection.query(
                                query_texts=[query],
                                n_results=n_results
                            )

            # Parse results
            retrieved_items = []

            if results is not None and results.get("documents"):
                documents = results["documents"]

                if documents and len(documents[0]) > 0:
                    result_len = len(documents[0])

                    for i in range(result_len):
                        doc_snippet = results["documents"][0][i]
                        metadata = results["metadatas"][0][i]
                        distance = results["distances"][0][i]

                        def get_snippet(text=doc_snippet):
                            return text

                        item = {
                            "source_url": metadata.get("url"),
                            "topic": metadata.get("topic"),
                            "get_context_snippet": get_snippet,
                            "distance": distance
                        }

                        retrieved_items.append(item)

            # If still no retrieved context, return safe fallback.
            if not retrieved_items:
                return [self.get_fallback_context(query)]

            return retrieved_items

        except Exception as e:
            print(f"⚠️ RAG retrieval failed for query '{query}'.")
            print(f"Error: {e}")

            return [self.get_fallback_context(query)]

    def get_fallback_context(self, query):
        """
        Safe fallback context used when Wikipedia or ChromaDB retrieval fails.
        This prevents the whole Streamlit app from crashing.
        """

        def fallback_snippet():
            return (
                "No external Wikipedia context was retrieved. "
                "Continue using the guardrail system without RAG context. "
                "Apply safety filtering, banned-word checks, toxicity checks, "
                "and LLM safety rewording based on the user's input."
            )

        return {
            "source_url": None,
            "topic": query,
            "get_context_snippet": fallback_snippet,
            "distance": None
        }

    def generate_hash_id(self, text):
        """
        Create an MD5 hash of the text content.
        This prevents the same data from being encoded multiple times.
        """

        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def handle_topic_ambiguity(self, topic) -> WikipediaPage:
        """
        Handles ambiguity in the ingestion process.

        If Wikipedia fails, this returns None instead of crashing the app.
        """

        try:
            page = wikipedia.page(topic, auto_suggest=True)
            return page

        except wikipedia.exceptions.DisambiguationError as e:
            best_match = e.options[0]

            print(f"⚠️ Ambiguous topic '{topic}'.")
            print(f"   ↳ Automatically switching to best match: '{best_match}'")
            print(f"   ↳ Other options found: {e.options[1:5]}")

            try:
                page = wikipedia.page(best_match, auto_suggest=False)
                return page

            except Exception as nested_error:
                print(f"❌ Failed to retrieve the auto-selected topic '{best_match}'.")
                print(f"Error: {nested_error}")
                return None

        except wikipedia.exceptions.PageError:
            print(f"⚠️ Wikipedia page not found for topic '{topic}'.")
            return None

        except requests.exceptions.JSONDecodeError as e:
            print(f"⚠️ Wikipedia returned invalid JSON for topic '{topic}'.")
            print(f"Error: {e}")
            return None

        except Exception as e:
            print(f"⚠️ Unexpected error while retrieving Wikipedia topic '{topic}'.")
            print(f"Error: {e}")
            return None

    def handle_topic_ambiguity_lists(self, topic) -> WikipediaPage:
        """
        Duplicate ambiguity handler kept for compatibility with older code.
        """

        return self.handle_topic_ambiguity(topic)

    class DISTANCE_RATING(Enum):
        """
        Enum for distance rating categories.
        Lower cosine distance means stronger relevance.
        """

        STRONG = 8
        GOOD = 5
        WEAK = 1
        IRRELEVANT = 0

    @staticmethod
    def has_strong_enough_relevance(distance_rating) -> bool:
        """
        Checks if the total distance rating is strong enough to be considered relevant.
        """

        return distance_rating >= ContextualizationEngine.DISTANCE_RATING.STRONG.value

    @staticmethod
    def interpret_distance(distance) -> DISTANCE_RATING:
        """
        Interprets cosine similarity distance between 0 and 1 as a rating.
        Lower distance means more relevant.
        """

        if distance < 0.20:
            return ContextualizationEngine.DISTANCE_RATING.STRONG
        elif distance < 0.45:
            return ContextualizationEngine.DISTANCE_RATING.GOOD
        elif distance < 0.60:
            return ContextualizationEngine.DISTANCE_RATING.WEAK
        else:
            return ContextualizationEngine.DISTANCE_RATING.IRRELEVANT
