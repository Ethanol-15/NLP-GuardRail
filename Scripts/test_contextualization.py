from modules.rag_engine import ContextualizationEngine

# 1. Initialize the Engine
rag = ContextualizationEngine()

# 2. Ingest Data (Optional step - only needed to build the DB)
   # these should handled externally to avoid making too many api calls to wikipedia (rate limit=500/hour)
# Let's add some knowledge about specific topics
# rag.ingest_wikipedia_topic("League of Legends")

print("------------------------------------------------")

# 3. The Query Loop
user_query = input("🔎 Insert Query:  ") # test some evil words here

# 4. Retrieve Context
context_results = rag.retrieve_context(user_query, ingest_if_needed=True)

# 5. Display Results (Simulating the output required)
if context_results:
    print(f"\nFound {len(context_results)} sources:\n")
    
    for idx, result in enumerate(context_results):
        url = result['source_url']
        # Calling the function inside the result to get the text
        snippet_text = result['get_context_snippet']()
        
        print(f"--- Result {idx + 1} ---")
        
        print(f"🔗 Link: {url}")
        print(f"📈 Distance: {result['distance']:.4f} ({ContextualizationEngine.interpret_distance(result['distance'])})") # this uses cosine similarity (could be configured)
        print(f"📄 Snippet: {snippet_text[:250]}...") # Truncated for display
        
        print("\n")
else:
    print("No relevant documents found.")