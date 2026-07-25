from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

code_example_1 = """
def compute_total(prices,taxes): 
    subtotal = sum(prices)
    return subtotal * (1 + taxes)
"""

code_example_2 = """
def get_sum(list, rate): 
    total = sum(list)
    return total * (1 + rate)
"""
code_example_3 = """
class ConnectionBD:
    def __init__(self, host, port):
        self.socket = None
        self.host = host
        self.port = port
"""
vec_a = embed_model.encode(code_example_1)
vec_b = embed_model.encode(code_example_2)
vec_c = embed_model.encode(code_example_3)

def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("A vs B (similar code):", cos_similarity(vec_a, vec_b))
print("A vs C (distinct code):", cos_similarity(vec_a, vec_c))