from modules.rag_engine import ContextualizationEngine

def __main__():
    """
    Creates a Vector Database (ChromaDB) then adds initial webtopics
    """
    rag = ContextualizationEngine(chroma_path="./chroma_db")

    harmful_topics = [
        "Tagalog profanity",
        "Philippine English language",
        "Bekimon",
        "Jejemon",
        "Pejorative",
        "Ethnic slur",
        "Political trolling in the Philippines",
        "Red-tagging in the Philippines",
        "War on drugs (Philippines)",
        "Filipino values",
        "Social issues in the Philippines",
        "Philippine politics",
        "Moro conflict"
    ]

    # https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/2
    general_knowledge = [
                        # Ambiguous Topics        
                        'Computer (device) Personal Computer',
                        'Gods (religion)',
                        'Electricity (physical phenomena)', 
                        'Ethics (moral phenomena)',
                        'Law (Legal)',
                        'Life (biological Process)',
                        'Logic (reasoning)',
                        'Matter (substance)',
                        'Moon (natural satellite)',
                        'Music (arrangement of sound)',
                        'North America (continent)',
                        'Oceania (geographical region)',
                        'Physics (scientific study)',
                        'Politics (decision making)',
                        'Science (study)',
                        'Sea (body of water)',
                        'The arts (creative disciplines)',
                        'Tool (object)',
                        'Trade (economy)',
                        'Universe (space)',
                        'Electricity (physical phenomena)', 
                        'Atmosphere of Earth (Air)',
                        'Asia (continent)', 
                        'Civilization (complex society)',
                        'Energy (scalar quantity)',

                        # easy questions
                        'Africa', 
                         'Agriculture', 
                         'Algebra', 
                         'Ancient history', 
                         'Animal', 
                         'Architecture', 
                         'Arithmetic', 
                         'Astronomy', 
                         'Biology', 
                         'Business', 
                         'Cell (biology)', 
                         'Chemical element', 
                         'Chemistry', 
                         'Climate', 
                         'Clothing', 
                         'Communication', 
                         'Culture', 
                         'Death', 
                         'Disease', 
                         'Early modern period', 
                         'Earth', 
                         'Ecology', 
                         'Economics', 
                         'Education', 
                         'Emotion', 
                         'Engineering', 
                         'Entertainment', 'Ethics', 'Ethnicity', 'Europe', 'Evolution', 'Family', 'Fire', 'Folklore', 'Food', 'Game', 'Geography', 'Geology', 'Geometry', 'Government', 'History', 'Home', 'Human', 'Human history', 'Human settlement', 'Human sexuality', 'Knowledge', 'Land', 'Language', 'Literature', 'Manufacturing', 'Mass media', 'Mathematics', 'Medicine', 'Mind', 'Modern era', 'Number', 'Performing arts', 'Philosophy', 'Plant', 'Post-classical history', 'Prehistory', 'Psychology', 'Religion', 'Society', 'Solar System', 'South America', 'State (polity)', 'Statistics', 'Sun', 'Technology', 'Time', 'Transport', 'Visual arts', 'War', 'Water', 'Writing']
    
    harmful_topics.extend(general_knowledge)
    rag.mass_ingest(harmful_topics)

    # rag.ingest_wikipedia_topic("Tagalog profanity")
    # rag.ingest_wikipedia_topic("Philippine English language")
    # rag.ingest_wikipedia_topic("Bekimon")
    # rag.ingest_wikipedia_topic("Jejemon")
    # rag.ingest_wikipedia_topic("Pejorative")
    # rag.ingest_wikipedia_topic("Ethnic slur")

    # rag.ingest_wikipedia_topic("Political trolling in the Philippines")
    # rag.ingest_wikipedia_topic("Red-tagging in the Philippines")
    # rag.ingest_wikipedia_topic("War on drugs (Philippines)")

    # rag.ingest_wikipedia_topic("Filipino values")
    # rag.ingest_wikipedia_topic("Social issues in the Philippines")
    # rag.ingest_wikipedia_topic("Philippine politics")
    # rag.ingest_wikipedia_topic("Moro conflict")

if __name__ == "__main__":
    __main__()