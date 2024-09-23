import os
import re
import yaml
import json
import traceback
from typing import *
from tqdm import tqdm
from bs4 import BeautifulSoup
from src.utils import setup_logger, split_sentences
from datasets import load_dataset
from joblib import Parallel, delayed

###### helper functions ######
def is_processed(dataset: str):
    """Check if the dataset is already processed.

    "Processed" means the raw dataset has been converted into 4 files:
    - queries.json: The queries dictionary.
    - answers.json: The answers dictionary.
    - docs.json: The docs list.
    - evidence.json: The evidence dictionary.

    Args:
        dataset (str): The dataset name.

    Returns:
        bool: True if the dataset is already processed, False otherwise.
    """
    queries_exists = os.path.exists(f"datasets/{dataset}/queries.json")
    answers_exists = os.path.exists(f"datasets/{dataset}/answers.json")
    docs_exists = os.path.exists(f"datasets/{dataset}/docs.json")
    evidence_exists = os.path.exists(f"datasets/{dataset}/evidence.json")
    return queries_exists and answers_exists and docs_exists and evidence_exists

def save_processed_data(queries, answers, docs, evidence, dataset: str):
    """Save the processed data.

    Args:
        queries (Dict): The queries dictionary.
        answers (Dict): The answers dictionary.
        docs (List): The docs list.
        evidence (Dict): The evidence dictionary.
        dataset (str): The dataset name.
    """
    if not os.path.exists("datasets"):              os.mkdir("datasets")
    if not os.path.exists(f"datasets/{dataset}"):   os.mkdir(f"datasets/{dataset}")
    json.dump(queries,  open(f"datasets/{dataset}/queries.json", "w"))
    json.dump(answers,  open(f"datasets/{dataset}/answers.json", "w"))
    json.dump(docs,     open(f"datasets/{dataset}/docs.json", "w"))
    json.dump(evidence, open(f"datasets/{dataset}/evidence.json", "w"))

def load_processed_data(dataset: str):
    """Load the processed data.

    Args:
        dataset (str): The dataset name.

    Returns:
        Tuple[Dict, Dict, Dict, Dict]: The queries, answers, docs, and evidence dictionaries.
    """
    queries =   json.load(open(f"datasets/{dataset}/queries.json", 'r'))
    answers =   json.load(open(f"datasets/{dataset}/answers.json", 'r'))
    docs =      json.load(open(f"datasets/{dataset}/docs.json", 'r'))
    evidence =  json.load(open(f"datasets/{dataset}/evidence.json", 'r'))
    return queries, answers, docs, evidence



###### dataloaders ######
def convert_ragbench_sents_to_tups(doc_sentences):
    """Convert the RAGBench sentences to tuples.

    In RAGBench, the sentences of each document are represented as a list of lists.
    Each inner list contains the sentence ID and the sentence text.
    Each sentence ID is labeled in the format: "{doc_id}{sent_id}".
    `doc_id` is numerical, whereas `sent_id` is alphabetical.
    `cache_alpha_to_num` is used to map the sentence ID to a numerical index for ease of use.

    Args:
        doc_sentences (List): The list of sentences in the document.

    Returns:
        List[Tuple[int, str]]: The list of sentence tuples, where each tuple contains the sentence index and the sentence text.
        Dict[str, int]: The mapping from alphabetical sentence ID to numerical index.
    """
    doc_repr_list = []          # store a new representation of the document
    cache_alpha_to_num = {}     # store a mapping from alphabetical sentence ID to numerical index

    for sent_idx, sentence in enumerate(doc_sentences):
        temp_sent = sentence.tolist()
        sent_id = re.sub(r'\d+', '', temp_sent[0])
        sent = ' '.join(temp_sent[1].split()).strip()
        doc_repr_list.append(tuple([sent_idx, sent]))
        cache_alpha_to_num[sent_id] = sent_idx

    return doc_repr_list, cache_alpha_to_num

def load_ragbench(dataset: str):
    """Load the RAGBench dataset.

    Source: https://huggingface.co/datasets/rungalileo/ragbench

    In RAGBench, there is no corpus. Instead, a list of documents is provided for each query.
    There can be duplicate documents across queries.
    Each document is already split into sentences.
    To create a corpus representation, we need to assign unique document IDs (i.e., indexes).
    The evidence is provided as a list of sentence IDs.

    Args:
        dataset (str): The dataset name.

    Returns:
        Dict[str, str]: A mapping from query ID to query text.
        Dict[str, str]: A mapping from query Id to answer text.
        List[List[str]]: A list of documents, where each document is a list of sentences.
        Dict[str, Dict[str, List[int]]]: A mapping from query ID to evidence dictionary,
            where each evidence dictionary maps document ID to a list of sentence IDs.
    """
    df = load_dataset("rungalileo/ragbench", dataset, split="test").to_pandas()
    df = df[df['all_relevant_sentence_keys'].apply(lambda x: len(x) > 0)]   # remove queries with no evidence
    df = df.drop_duplicates(subset=['id'])                                  # remove duplicate queries
    df.sort_values(by='id', inplace=True)                                   # sort by ID to ensure consistency

    # initialize the return variables
    queries = {}
    answers = {}
    docs = []
    evidence = {}

    # keep track of unique documents and their assigned IDs
    doc_to_id = {}
    doc_id_counter = 0

    # iterate through each row of the dataframe
    for _, row in df.iterrows():
        query_id = row['id']
        query = row['question']
        answer = row['response']
        documents_sentences = row['documents_sentences']
        relevant_sentence_keys = row['all_relevant_sentence_keys']

        # map query ID to query text and answer text
        queries[query_id] = query
        answers[query_id] = answer

        # initialize evidence for this query
        evidence[query_id] = {}

        # process each document
        for doc_idx, doc_sentences in enumerate(documents_sentences):
            # get the list of sentences for the current document
            doc_sentences = documents_sentences[doc_idx].tolist()

            # Create a unique representation of the document to avoid duplicates
            doc_repr_list, cache_alpha_to_num = convert_ragbench_sents_to_tups(doc_sentences)

            # tuple representation of the document as the key for the doc_to_id dictionary
            doc_repr = tuple(doc_repr_list)

            # if this document has not been assigned a unique ID, assign it
            if doc_repr not in doc_to_id:
                doc_to_id[doc_repr] = doc_id_counter
                docs.append([tup[1] for tup in doc_repr_list])
                doc_id_counter += 1

            # get the assigned document ID
            doc_id = doc_to_id[doc_repr]

            # initialize the evidence for this document
            evidence[query_id][doc_id] = []

            # process evidence sentence IDs for this document
            for sentence_id in relevant_sentence_keys:
                # check if the sentence ID is from this document
                if any(sentence_id == sent[0] for sent in doc_sentences):
                    # add a tuple of (document ID, sentence ID) to the evidence list
                    evidence[query_id][doc_id].append(cache_alpha_to_num[sentence_id.replace(str(doc_idx), '')])

            # remove the document ID if there is no evidence
            if evidence[query_id][doc_id] == []:
                del evidence[query_id][doc_id]

    return queries, answers, docs, evidence

def load_qasper():
    """Load the QASPER dataset.

    Source: https://huggingface.co/datasets/allenai/qasper

    In QASPER, the documents are provided as a list of sections and paragraphs.
    There are multiple questions for each document.
    The answers are provided as free-form text.
    The evidence is provided as a list of sentences.
    To match the format of the other datasets, we process the evidence to map to a list of sentence IDs.

    Returns:
        Dict[str, str]: A mapping from query ID to query text.
        Dict[str, str]: A mapping from query Id to answer text.
        List[List[str]]: A list of documents, where each document is a list of sentences.
        Dict[str, Dict[str, List[int]]]: A mapping from query ID to evidence dictionary,
            where each evidence dictionary maps document ID to a list of sentence IDs.
    """
    # initialize the return variables
    queries = {}
    answers = {}
    docs = []
    evidence = {}

    # load the QASPER dataset
    dataset = load_dataset("allenai/qasper", split='test')

    # iterate through each row of the dataset
    for doc_id, row in tqdm(enumerate(dataset)):
        # get the full academic paper text
        doc = row['abstract']
        for section in row['full_text']['paragraphs']:
            for paragraph in section:
                doc += '/n/n'
                doc += paragraph
        sents = split_sentences(doc)
        docs.append(sents)

        # process each question-answer pair for this document
        for i, question in enumerate(row['qas']['question']):
            query_id = row['qas']['question_id'][i]
            answer = row['qas']['answers'][i]['answer'][0]
            if answer['unanswerable'] or answer['free_form_answer'].strip() == '':
                continue
            queries[query_id] = question.strip().replace('?','') + " in " + row['title'] + '?'
            answers[query_id] = answer['free_form_answer']

            # process the evidence for this question
            evidence[query_id] = {}
            evidence[query_id][str(doc_id)] = []
            evidence_sents = set(split_sentences('\n'.join(answer['evidence'])))
            for sent_id, sent in sents:
                if sent in evidence_sents:
                    evidence[query_id][str(doc_id)].append(sent_id)

    return queries, answers, docs, evidence

def load_conditionalqa():
    """Load the ConditionalQA dataset.

    Source: https://haitian-sun.github.io/conditionalqa/

    In ConditionalQA, the documents are provided in the HTML format.

    Returns:
        Dict[str, str]: A mapping from query ID to query text.
        Dict[str, str]: A mapping from query Id to answer text.
        List[List[str]]: A list of documents, where each document is a list of sentences.
        Dict[str, Dict[str, List[int]]]: A mapping from query ID to evidence dictionary,
            where each evidence dictionary maps document ID to a list of sentence IDs.
    """
    # initialize the return variables
    queries = {}
    answers = {}
    docs = []
    evidence = {}

    # keep track of unique documents and their assigned IDs
    url_to_id = {}

    # load and process the original documents to the desired format
    docs_original = json.load(open('datasets/conditionalqa/original/documents.json','r'))
    for i,doc in tqdm(enumerate(docs_original), total=len(docs_original)):
        url_id = doc['url'].split('/')[-1]
        url_to_id[url_id] = i
        doc_new = doc['title']+'\n'+'\n'.join([BeautifulSoup(sent, "html.parser").get_text() for sent in doc['contents']])
        sents = split_sentences(doc_new)
        docs.append(sents)

    # load and process the original questions and answers to the desired format
    qas = json.load(open('datasets/conditionalqa/original/dev.json','r'))
    for qa in tqdm(qas, total=len(qas)):
        if qa['not_answerable']: continue                           # skip the questions with no answer
        url_id = qa['url'].split('/')[-1]
        if url_id not in url_to_id: continue                        # skip the questions with no matching document
        doc_id = url_to_id[url_id]
        curr_sents = docs[doc_id]
        query_id = qa['id']
        queries[query_id] = qa['scenario'] + '\n' + qa['question']  # concatenate the scenario and question
        answers[query_id] = qa['answers'][0][0]                     # get the first answer as the answer

        # process the evidence for this question
        evidence[query_id] = evidence.get(query_id, {})
        evidence[query_id][str(doc_id)] = evidence[query_id].get(str(doc_id), [])
        processed_evidence = set(split_sentences('\n'.join([BeautifulSoup(sent, "html.parser").get_text() for sent in qa['evidences']])))
        for sent_id, sent in enumerate(curr_sents):
            if sent in processed_evidence:
                evidence[query_id][str(doc_id)].append(sent_id)

    return queries, answers, docs, evidence

def load_data(dataset: str):
    """Load the dataset.

    Args:
        dataset (str): The dataset name.

    Returns:
        Tuple[Dict, Dict, Dict, Dict]: The queries, answers, docs, and evidence dictionaries.
    """
    # if the dataset is processed, load the processed data
    if is_processed(dataset):
        return load_processed_data(dataset)

    # if the dataset is not processed and comes from ragbench, process it from ragbench
    if dataset in {'cuad', 'delucionqa', 'emanual', 'expertqa', 'msmarco', 'pubmedqa', 'covidqa', 'hagrid', 'hotpotqa', 'techqa'}:
        return load_ragbench(dataset)
    elif dataset == "qasper":
        return load_qasper()
    elif dataset == "conditionalqa":
        return load_conditionalqa()



###### MAIN FUNCTION ######
def load_and_save(dataset: str):
    """Load and save the dataset.

    Args:
        dataset (str): The dataset name.
    """
    logger = setup_logger()
    try:
        logger.info(f"Loading {dataset}...")
        queries, answers, docs, evidence = load_data(dataset)
        logger.info(f"Saving {dataset}...")
        save_processed_data(queries, answers, docs, evidence, dataset)
    except Exception as e:
        logger.info(f"Error occurred for dataset {dataset}: {e}")
        traceback.print_exc()



# Executing this script will load the datasets and save the processed data.
if __name__ == "__main__":
    cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    Parallel(n_jobs=4)(delayed(load_and_save)(dataset=dataset) for dataset in cfg['datasets'])