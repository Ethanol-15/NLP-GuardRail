# main.py
from modules.rag_engine import ContextualizationEngine

# 1. Initialize the Engine
rag = ContextualizationEngine()

# 2. Ingest Data (Optional step - only needed to build the DB)
# Let's add some knowledge about specific topics
rag.ingest_wikipedia_topic("Philippines") 
rag.ingest_wikipedia_topic("Duterte")

print("------------------------------------------------")

# 3. The Query Loop
user_query = "What is the current political situation in the Philippines?"

print(f"🔎 User Query: {user_query}")

# 4. Retrieve Context
context_results = rag.retrieve_context(user_query)

# 5. Display Results (Simulating the output required)
if context_results:
    print(f"\nFound {len(context_results)} relevant sources:\n")
    
    for idx, result in enumerate(context_results):
        url = result['source_url']
        # Calling the function inside the result to get the text
        snippet_text = result['get_context_snippet']() 
        
        print(f"--- Result {idx + 1} ---")
        print(f"🔗 Link: {url}")
        print(f"📄 Snippet: {snippet_text[:200]}...") # Truncated for display
        print("\n")
else:
    print("No relevant documents found.")