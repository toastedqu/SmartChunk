import json
import nltk
import pickle
import evaluate
import numpy as np
from sentence_transformers import util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.encoder import Encoder
from src.utils import create_hyperparam_str, get_configs, setup_logger

LOGGER = setup_logger()
BERTSCORE = evaluate.load('bertscore')
BLEU_SF = SmoothingFunction().method1
ENCODER = Encoder(device=2)
nltk.download('punkt')
nltk.download('punkt_tab')

def jsonl_to_dict(dataset, fname):
    generated_answers = {}
    with open(f'results/{dataset}/generated_answers/{fname}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            query_id = data.pop('query_id')
            generated_answers[query_id] = data
    return generated_answers

def get_bleu_per_pair(ref, gen):
    ref_tokens = nltk.word_tokenize(ref)
    gen_tokens = nltk.word_tokenize(gen)
    return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=BLEU_SF)

def get_bleu(refs, gens):
    if isinstance(refs, str):
        return get_bleu_per_pair(refs, gens)
    return np.mean([get_bleu_per_pair(ref, gen) for ref, gen in zip(refs, gens)])

def get_bertscore(refs, gens):
    if isinstance(refs, str):
        refs, gens = [refs], [gens]
    return np.mean(BERTSCORE.compute(predictions=gens, references=refs, model_type="microsoft/deberta-xlarge-mnli", verbose=False)['f1'])

def get_qa_cos_sim(queries, answers):
    queries_embeddings = ENCODER.encode_queries(queries)
    answers_embeddings = ENCODER.encode_documents(answers)
    return np.mean([util.cos_sim(q,a).item() for q,a in zip(queries_embeddings, answers_embeddings)])

def evaluate_generation(cfg):
    dataset = cfg['dataset']
    hyperparams = cfg['hyperparams']
    fname = create_hyperparam_str(hyperparams)

    queries = json.load(open(f'datasets/{dataset}/sampled_queries.json', 'r'))
    answers = json.load(open(f'datasets/{dataset}/answers.json', 'r'))
    generated_answers = jsonl_to_dict(dataset, fname)

    query_ids = list(sorted(list(queries.keys())))
    l_queries = [queries[query_id] for query_id in query_ids]
    l_answers = [answers[query_id] for query_id in query_ids]
    l_generated_answers = [generated_answers[query_id]['answer'] for query_id in query_ids]

    bleu = get_bleu(l_answers, l_generated_answers)
    bertscore = get_bertscore(l_answers, l_generated_answers)
    qa_sim = get_qa_cos_sim(l_queries, l_generated_answers)

    return bleu, bertscore, qa_sim

if __name__ == "__main__":
    results = {}
    cfgs = get_configs()
    for cfg in cfgs:
        dataset = cfg['dataset']
        hyperparams = cfg['hyperparams']
        fname = create_hyperparam_str(hyperparams)
        if dataset not in results:
            results[dataset] = {}
        bleu, bertscore, qa_sim = evaluate_generation(cfg)
        results[dataset][fname] = {
            'bleu': bleu,
            'bertscore': bertscore,
            'qa_cos_sim': qa_sim
        }
    with open(f"results/results_generation.pkl", "wb") as f:
        pickle.dump(results, f)