# SmartChunk

Motivation: Chunking is a step in RAG. However, currently, chunk sizes are usually determined arbitrarily, either making a vector DB inefficiently used or reducing the RAG precision.

Idea: Instead of chunking in sentences or in paragraphs, we chunk in a smarter way: split every document into chunks that are semantically similar by using clusters as chunks from hierarchical clustering, since there is a case that only a portion of a document is relevant to a query.

Modules:
- Benchmark loaders. A loader will load test data from an IR/RAG benchmark. It needs to support a variety of benchmarks. (BEIR)
- Chunkers. A Python function. Given a string, returns a list of strings. We store the chunks into a vector DB for retrieval.
- Embed each chunk (per chunker, per embedder, per dataset) and save all embeddings in a way that it can be easily accessed. Also an easy way to locate the textual data corresponding to an embedding vector is also needed. It can be as simple as a huge numpy array. We don't care about speed here. Check how big the dataset is. If it cannot fit into the RAM (say 16GB), we need a dedicated vector DB. (Now we just use memory)
- A query-and-match function. Its input is: a query (string), an identifier to the storage of the embeddings (a string, for example). Return: a string or a list of strings, depending on the benchmark and metric, e.g., precision@n. (BEIR has the case without vector DB)
- IR/RAG metrics. A Python function, tells how accurate the retrieved string is compared with the ground truth. (BEIR)

BEIR: https://github.com/beir-cellar/beir
