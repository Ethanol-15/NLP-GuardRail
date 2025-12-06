from modules.rag_engine import ContextualizationEngine

def __main__():
    """
    Creates a Vector Database (ChromaDB) then adds initial webtopics
    """
    rag = ContextualizationEngine(chroma_path="./chroma_db")

    rag.ingest_wikipedia_topic("Tagalog profanity")
    rag.ingest_wikipedia_topic("Philippine English language")
    rag.ingest_wikipedia_topic("Bekimon")
    rag.ingest_wikipedia_topic("Jejemon")
    rag.ingest_wikipedia_topic("Pejorative")
    rag.ingest_wikipedia_topic("Ethnic slur")

    rag.ingest_wikipedia_topic("Political trolling in the Philippines")
    rag.ingest_wikipedia_topic("Red-tagging in the Philippines")
    rag.ingest_wikipedia_topic("War on drugs (Philippines)")

    rag.ingest_wikipedia_topic("Filipino values")
    rag.ingest_wikipedia_topic("Social issues in the Philippines")
    rag.ingest_wikipedia_topic("Philippine politics")
    rag.ingest_wikipedia_topic("Moro conflict")

if __name__ == "__main__":
    __main__()