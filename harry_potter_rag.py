import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 환경변수에서 Gemini 키 로드
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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

def generate_response(character, question, retrieved_scenes):

    # 프롬프트 구성 (이전과 동일)
    prompt = f"""
    You are the character "{character}" from the Harry Potter universe.
    You are highly knowledgeable about Hogwarts School of Witchcraft and Wizardry, magical laws, the Ministry of Magic, the Dark Arts, Quidditch, and other aspects of the wizarding world.
    Please answer the following question based on your character's personality, knowledge, and experience.

    Below are reference scenes related to the question. These are actual events that happened in your life. Whenever possible, use these references to provide your answer.
    If the reference scenes are too long, they may be summarized versions of the original events.

    Reference Scenes:
    {retrieved_scenes}

    Question: {question}
    """

    response = model.generate_content(prompt)

    return response.text.strip()
