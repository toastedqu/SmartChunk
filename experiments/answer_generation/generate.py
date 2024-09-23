import os
import json
import pickle
import tiktoken
from tqdm import tqdm
from openai import OpenAI
from src.utils import create_hyperparam_str, get_configs, setup_logger

LOGGER = setup_logger()
MODEL_NAME = "gpt-4o-mini"
CLIENT = OpenAI()
TOKEN_COUNTER = tiktoken.encoding_for_model(MODEL_NAME)
PROMPT_TEMPLATE = """Generate an accurate answer for the query solely based on the provided passages below. You must only use information from the provided passages.

Query: {query}

Passages:
{passages}
"""

def count_tokens(text):
    """Count the number of tokens in the text.

    Args:
        text (str): The text to count the tokens of.

    Returns:
        int: The number of tokens in the text."""
    return len(TOKEN_COUNTER.encode(text))

def generate(prompt, max_tokens=400):
    """Generate an answer based on the prompt.

    Args:
        prompt (str): The prompt to generate the answer from.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 400.

    Returns:
        str: The generated answer.
    """
    return CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0
    ).choices[0].message.content.strip()

def save_retrieved_sents(cfg):
    """Save the retrieved sentences for each query in the configuration.

    Args:
        cfg (dict): The configuration to save the retrieved sentences for.
    """
    dataset = cfg['dataset']
    hyperparams = cfg['hyperparams']
    fname = create_hyperparam_str(hyperparams)

    # sort scores
    sorted_scores = pickle.load(open(f'results/{dataset}/scores_{fname}.pkl', 'rb'))
    sorted_scores = {query_id: sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10] for query_id, scores in sorted_scores.items()}

    # load chunks and docs
    chunks = json.load(open(f'chunks/{dataset}/{fname}.json', 'r'))
    docs = json.load(open(f'datasets/{dataset}/docs.json', 'r'))

    # get retrieved sentences
    retrieved_sents = {}
    for query_id, scores in sorted_scores.items():
        retrieved_sents[query_id] = []
        for doc_chunk_id, _ in scores:
            doc_id = int(doc_chunk_id.split('|')[0])
            selected_sent_ids = chunks[doc_id][doc_chunk_id]
            cache_sents = [docs[doc_id][sent_id] for sent_id in selected_sent_ids]
            retrieved_sents[query_id].append(cache_sents)

    # save retrieved sentences
    if not os.path.exists(f'results/{dataset}/retrieved_sents'):
        os.mkdir(f'results/{dataset}/retrieved_sents')
    json.dump(retrieved_sents, open(f'results/{dataset}/retrieved_sents/{fname}.json', 'w'))

def generate_per_cfg(cfg, k=1):
    dataset = cfg['dataset']
    hyperparams = cfg['hyperparams']
    fname = create_hyperparam_str(hyperparams)

    # load retrieved sentences and queries
    retrieved_sents = json.load(open(f'results/{dataset}/retrieved_sents/{fname}.json', 'r'))
    retrieved_sents = {query_id: retrieved_sents[query_id][:k] for query_id in retrieved_sents} # only use top k retrieved sentences
    queries = json.load(open(f'datasets/{dataset}/sampled_queries.json', 'r'))

    # create generated answers directory
    if not os.path.exists(f'results/{dataset}/generated_answers'):
        os.mkdir(f'results/{dataset}/generated_answers')

    # generate answers
    for query_id, query in queries.items():
        generated_answers = {}

        # concatenate passages
        passages = '\n\n'.join(['\n'.join(cache_sents) for cache_sents in retrieved_sents[query_id]])

        # generate answer
        prompt = PROMPT_TEMPLATE.format(query=query, passages=passages)
        generated_answer = generate(prompt)

        # store generated answer
        generated_answers['query_id'] = query_id
        generated_answers['answer'] = generated_answer
        generated_answers['passages_token_count'] = count_tokens(passages)
        generated_answers['answer_token_count'] = count_tokens(generated_answer)

        # log
        LOGGER.info(f"DATASET: {dataset}")
        LOGGER.info(f"QUERY {query_id}: {query}")
        LOGGER.info(f"ANSWER: {generated_answer}")

        # save generated answers
        with open(f'results/{dataset}/generated_answers/{fname}.jsonl', 'a') as f:
            f.write(json.dumps(generated_answers) + '\n')

if __name__ == "__main__":
    cfgs = get_configs()
    for cfg in tqdm(cfgs):
        save_retrieved_sents(cfg)
    for cfg in tqdm(cfgs):
        generate_per_cfg(cfg)