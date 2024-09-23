# SmartChunk
This project evaluates the performance of different chunking strategies for RAG (Retrieval-Augmented Generation).

In a RAG system, documents are divided into chunks. The system calculates relevance scores between each chunk and a user query, retrieves the most relevant chunks, and passes them to a generative language model to produce a response. Both the content and size of the chunks impact retrieval and generation performance, as demonstrated by [Chen et al. (2023)](https://arxiv.org/abs/2312.06648) and [Wadhwa et al. (2024)](https://arxiv.org/abs/2406.12824).

Since ground-truth chunks or query-chunk relevance scores are unavailable, it is challenging to directly assess the quality of retrieved chunks for a given query. To approximate the quality, we performed 3 experiments:
1. **Document Retrieval**: Each retrieved chunk is mapped to its source document, and the results are compared with ground-truth query-document relevance scores.
2. **Evidence Retrieval**: Each retrieved chunk is mapped to its corresponding evidence, and the results are compared with ground-truth query-evidence relevance scores.
3. **Generation**: Queries and retrieved chunks are fed into an LLM to generate answers, which are then compared to ground-truth answers.

This page only covers technical details. See [Notion](https://www.notion.so/vectara/SmartChunk-076aa4f2f2384331a2e2c4a6ea551e0e) for other info.

# Modules
## Src
The `src/` folder contains the main files that can be used outside the experiments.
- `chunkers.py`: A collection of chunking strategies.
- `encoder.py`: A custom encoder class as an interface for the `sentence_transformers` package.
- `utils.py`: A collection of helper functions used throughout the project.

## Datasets
The `datasets/` folder contains the datasets used in the experiments. Each subfolder corresponds to a specific dataset and consists of the following files:
- `queries.json`: A collection of queries.
- `docs.json`: A collection of documents.

The following files and folders are optional:
- `qrels.json`: A mapping of "query_id -> doc_id -> score", where the score is the ground-truth relevance score between the query and the document.
- `answers.json`: A mapping of "query_id -> answer".
- `evidence.json`: A mapping of "query_id -> doc_id -> list of sent_ids", where each sent_id corresponds to a ground-truth evidence sentence.
- `original/`: A folder of the original data.

Please check Datasets for details on formatting and stuff.

## Chunks
The `chunks/` folder stores chunks from all chunker configurations for all datasets. Each subfolder corresponds to a specific dataset and consists of JSON files, where each JSON file stores the chunks obtained from a specific chunker configuration. The format of each JSON file is a mapping from a "doc_id|chunk_id" string to a list of sent_ids, where each sent_id corresponds to a sentence in the document. The naming of each JSON file is as follows:

### Configuration String Format
We represent each chunker configuration in the format below:

```hyperparam_str = "{chunker_name}|{hyperparam_name_1}_{hyperparam_value_1}|{hyperparam_name_2}_{hyperparam_value_2}|..."```

where
- Chunker name is separated from hyperparameters by '|'.
- Hyperparameter name-value pairs are separated from each other by '|'.
- The name and value for each hyperparameter are separated by '_'.
- The decimal point in a float value is replaced by '-' for filename compatibility.

For example, a `PositionalChunker` with `n_chunks = 7` and `n_sents_overlap = 1` is named as:

```PositionalChunker|n_chunks_7|n_sents_overlap_1.json```

## Embeddings
The `embeddings/` folder contains all the precomputed embeddings necessary for the final evaluations. Each subfolder corresponds to a specific dataset and consists of the following files:
- `queries_embeddings.pkl`: A mapping of "query_id -> query_embedding".
- `sents_embeddings.pkl`: A mapping of "doc_id -> list of sentence_embeddings", where each sentence embedding corresponds to the sentence at the exact index in the document.
- `chunk_embeddings/`: A folder of chunk embeddings for all chunker configurations for all datasets. Each subfolder corresponds to a specific dataset and consists of Pickle files. The format of each Pickle file is a mapping from a "doc_id|chunk_id" string to the corresponding chunk embedding. The naming of each Pickle file follows the configuration string format above.

## Results
The `results/` folder contains all experimental results. Each subfolder corresponds to a specific dataset and consists of the following files:
- `results_document.json`: A mapping of "hyperparam_str -> k -> metric -> score", where `k` is the number of top chunks being retrieved.
- `results_evidence.json`: A mapping of "hyperparam_str -> k -> metric -> score".
- `results_generation.json`: A mapping of "hyperparam_str -> k -> metric -> score".
- `sim_scores/`: A folder of all query-chunk similarity scores. Each Pickle file in this folder corresponds to a specific chunker configuration and is a mapping of "query_id -> "doc_id|chunk_id" -> sim_score".
- `retrieved_sents/`: A folder of retrieved sentences from all chunker configurations for all queries. Each JSON file maps a query_id to a list of passages ready to be fed to an LLM for the answer generation experiment. Each passage consist of a list of sentences.
- `generated_answers/`: A folder of LLM-generated answers based on the retrieved sentences. Each JSONL file contains the generated answers for all queries. Each line is a dictionary with the following key-value pairs:
    - `query_id`: The query ID.
    - `answer`: The LLM-generated answer.
    - `passage_token_count`: The number of tokens in the passages fed to the LLM, for this query.
    - `answer_token_count`: The number of tokens in the LLM-generated answer, for this query.

## Experiments
The `experiments/` folder contains the scripts used for all 3 experiments. Each subfolder corresponds to a specific experiment.

### Document retrieval
The `document_retrieval/` folder consists of (tbd)

### Evidence retrieval
The `evidence_retrieval/` folder consists of 5 files:
- `load_data.py`: loads and processes all raw datasets into datasets of our desired format.
- `precompute_embeddings.py`: precomputes the embeddings for queries and documents for all datasets.
- `chunk_data.py`: chunks all documents in all datasets with all chunker configurations.
- `evaluate.py`: evaluates evidence retrieval performance for all chunker configurations across all datasets.
- `results_analysis.ipynb`: analyzes the performance of each chunker configuration on each dataset for each metric, for evidence retrieval.

### Answer generation
The `answer_generation/` folder assumes that chunks and chunk embeddings are already precomputed from previous experiments. It consists of 3 files:
- `generate.py`: generates an answer for each query-chunk pair for each chunker for each dataset.
- `evaluate.py`: evaluates answer generation for all chunker configurations across all dataset.
- `results_analysis.ipynb`: analyzes the performance of each chunker configuration on each dataset for each metric, for answer generation.

## Others
- `config.yaml`: consists of the configurations for all datasets, encoders, chunkers, and all hyperparameter configurations for each chunker.
- `requirements.txt`: consists of all high-level packages used in this project.
- `misc/`: consists of all minor files which address issues or cache occurred during the project.

<!-- ## Replicate
Create a virtual environment (Python 3.10 preferred). Install all the required dependencies via
```
pip install -r requirements.txt
```

Then simply execute `run.py`.
```
python run.py
```

## Customization
### Custom Dataloader
To test on your own data, simply put your dataloader function in `dataloader.py`, add the function call to the `load_data(dataset)` function following the format of the commented lines, and add your dataset name to the `datasets` list in `config.yaml`.

Your custom dataloader function must return the following values:
- `corpus`: A dictionary mapping document IDs to dictionaries. Each sub-dictionary can only have the following 2 mappings:
    - `"title"`: (optional) The title of the document.
    - `"text"`: The document content.
- `queries`: A dictionary mapping query IDs to query strings.
- `qrels`: A dictionary mapping query IDs to corresponding document-score dictionaries. Each document-score dictionary maps document IDs to corresponding document scores. You do NOT need to include all documents in the corpus for each query. Only the relevant ones matter.

The returned values of your dataloading function must strictly follow the [BEIR format](https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset):
```
corpus = {
    "doc1" : {
        "title": "Albert Einstein",
        "text": "Albert Einstein was a German-born theoretical physicist. who developed the theory of relativity, \
                 one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for \
                 its influence on the philosophy of science. He is best known to the general public for his mass–energy \
                 equivalence formula E = mc2, which has been dubbed 'the world's most famous equation'. He received the 1921 \
                 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law \
                 of the photoelectric effect', a pivotal step in the development of quantum theory."
        },
    "doc2" : {
        "title": "", # Keep title an empty string if not present
        "text": "Wheat beer is a top-fermented beer which is brewed with a large proportion of wheat relative to the amount of \
                 malted barley. The two main varieties are German Weißbier and Belgian witbier; other types include Lambic (made\
                 with wild yeast), Berliner Weisse (a cloudy, sour beer), and Gose (a sour, salty beer)."
    },
}

queries = {
    "q1" : "Who developed the mass-energy equivalence formula?",
    "q2" : "Which beer is brewed with a large proportion of wheat?"
}

qrels = {
    "q1" : {"doc1": 1},
    "q2" : {"doc2": 1},
}
```

### Custom Chunker
To test your custom chunker, simply put your chunker function in `chunkers.py`, and add the chunker name (same as its function name) and its hyperparameter configurations to `config.yaml`.

For structure and readability, if your chunker function depends on other clustering algorithms, please put them in `clusters.py`. If your chunker function depends on other helper functions, please put them in `utils.py`.

Your custom chunker function must meet the following requirements:
1. Each chunker must have the following arguments:
    - `doc (str)`: The document text
    - `doc_id (str)`: The document ID

    Feel free to add kwargs on your own.
2. Each chunker must return the following lists of strings:
    - `texts (List[str])`: The list of chunked texts
    - `ids (List[str])`: The list of chunk IDs
3. Each returned chunk ID must follow the `{doc_id}|{chunk_id}` format.

### Custom Encoder
This projects uses [Langchain](https://python.langchain.com/v0.2/docs/integrations/text_embedding/) to access various embeddings.

To test a new encoder on Langchain, simply add the Langchain embedding to `get_encoder(encoder_name)` in `encoder.py` following the format of the commented lines, and add your encoder name to the `encoders` list in `config.yaml`.

To test your custom encoder that is not on Langchain, please follow [Langchain Custom Embeddings](https://api.python.langchain.com/en/latest/chains/langchain.chains.hyde.base.HypotheticalDocumentEmbedder.html) to create your own encoder class, put it in `encoder.py`, and add it to the `get_encoder(encoder_name)` function, so that the codes can utilize your model smoothly.

### Custom Experiment
To run the experiment with your new configurations, make sure in `config.yaml`:
- Your dataset is in `datasets` list.
- Your encoder is in `encoders` list.
- Your chunker is in `chunkers` list.
- Your hyperparameter configuration is appended at the bottom, following the format of the commented lines.
- Everything you don't need is commented out.

write a loop in the `run.py` following the format of the commented lines. -->