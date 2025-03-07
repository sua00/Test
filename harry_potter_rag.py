import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# .env 파일 읽기
load_dotenv()

# 환경변수에서 Gemini 키 로드
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# 임베딩 모델 및 FAISS 인덱스 로드 (사전 생성 필요)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# 저장된 데이터 로드 (검색용)
scenes = np.load("scene_texts.npy", allow_pickle=True).tolist()
index = faiss.read_index("dialogue_index.faiss")

def search_similar_scenes(query, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)

    results = []
    for idx in I[0]:
        results.append(scenes[idx])

    return results

def generate_response(character, role, question, retrieved_scenes):
    import google.generativeai as genai
    import os

    prompt = f"""
    You are the character "{character}" from the Harry Potter universe.
    Right now, you are talking to someone who is your "{role}".
    Please respond to this "{role}" in the tone and style you would naturally use when talking to a {role}.

    You trust this {role}, so you can speak naturally — whether it's being respectful to a professor, casual with a friend, or warm with a family member.

    You can refer to the scenes below if they help you remember something, but you don't need to quote them directly.

    Reference Scenes (for your memory only):
    {retrieved_scenes}

    The {role}'s question:
    {question}
    """

    response = model.generate_content(prompt)

    return response.text.strip()

