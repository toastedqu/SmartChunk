import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, util

class Encoder:
    def __init__(self, model_name: str = "dunzhang/stella_en_1.5B_v5", device: int = 0):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=f"cuda:{device}")
        self.prompt_name = "s2p_query"

    def encode_queries(self, queries: str | List[str]) -> np.ndarray:
        if isinstance(queries, str): queries = [queries]
        return self.model.encode(queries, prompt_name=self.prompt_name)

    def encode_documents(self, docs: str | List[str]) -> np.ndarray:
        if isinstance(docs, str): docs = [docs]
        return self.model.encode(docs)

if __name__ == "__main__":
    encoder = Encoder()
    queries = [
        'are judo throws allowed in wrestling?',
        'how to become a radiology technician in michigan?'
    ]
    docs = [
        "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan.",
        "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
    ]
    queries_embeddings = encoder.encode_queries(queries)
    docs_embeddings = encoder.encode_documents(docs)

    sim_scores_cos = util.cos_sim(queries_embeddings, docs_embeddings)
    sim_scores_dot = util.dot_score(queries_embeddings, docs_embeddings)

    print(sim_scores_cos)
    print(sim_scores_dot)