import pathlib, os
import functools
import argparse

from beir import util

import datasets

import dataloader
import chunker
import retrieve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA-QA")
    parser.add_argument("-d", "--dataset", type=str, dest="dataset", default="scifact", help="dataset name in BEIR")
    parser.add_argument("-s", "--split", type=str, dest="split", default="test", help="split name in BEIR")
    parser.add_argument("-c", "--chunker", type=str, dest="chunker", default="chunker.whole_chunker", help="chunker function")
    parser.add_argument("-k", "--top_k", type=str, dest="topk", default="[5, 20]", help="top k retrieval")
    args = parser.parse_args()

    print("BEIR-IR")
    print("Arguments:", args)

    # download and unzip the dataset
    dataset = args.dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    # load and embed the dataset
    corpus = dataloader.read_corpus(data_path)
    qrels = dataloader.read_qrels(data_path, split=args.split)
    queries = dataloader.read_queries(data_path, qrels)

    queries_beir = dataloader.convert_queries(queries)
    qrels_beir = dataloader.convert_qrels(qrels)

    corpus_len = len(corpus)

    # chunk the documents by the selected chunker
    selected_chunker = eval(args.chunker)
    corpus_chunked = corpus.map(functools.partial(dataloader.beir_process, chunker=selected_chunker), batched=True, batch_size=1)

    # filter out the original documents, only keep the split documents
    corpus_split: datasets.Dataset = corpus_chunked.filter(lambda row: row["origin"] != "corpus")
    corpus_split_beir = dataloader.convert_corpus(corpus_split)

    retrieve.retrieve_and_evaluate(corpus_split_beir, queries_beir, qrels_beir, k_values=eval(args.topk))
