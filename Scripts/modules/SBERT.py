from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(context, response):
    # Encode sentences
    embeddings = model.encode([context, response])
    
    # Compute cosine similarity
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()