# SmartChunk
This project aims to investigate the performance of various chunking strategies for document retrieval.

## Modules
- `dataloader.py`: Functions for loading different datasets.
- `chunkers.py`: Functions for using different chunking strategies.
- `encoder.py`: A script with 2 parts:
    - `get_encoder(encoder_name)`: A function to get Langchain encoders for semantic chunking
    - `CustomRetriever`: A class to use Langchain encoders for document retrieval.
- `run.py`: The full experiment. The evaluation results are stored in `results/results.jsonl`, where each line consists of:
    - Metrics & Scores
    - Dataset name
    - Encoder name
    - Chunker name
    - Hyperparameters for the chunker (if none, empty dictionary)
- `utils.py`: Helper functions
- `clusters.py`: Functions for clustering-based chunking strategies.

## Replicate
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

write a loop in the `run.py` following the format of the commented lines.