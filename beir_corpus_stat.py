import pathlib, os
import spacy
import numpy as np
import plotnine as p9

from beir import util

import dataloader

nlp = spacy.load("en_core_web_sm")

datasets = ["msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", "webis-touche2020", "cqadupstack.android", "cqadupstack.english", "cqadupstack.gaming", "cqadupstack.gis", "cqadupstack.mathematica", "cqadupstack.physics", "cqadupstack.programmers", "cqadupstack.stats", "cqadupstack.tex", "cqadupstack.unix", "cqadupstack.webmasters", "cqadupstack.wordpress", "quora", "dbpedia-entity", "scidocs", "fever", "climate-fever", "scifact"]


def __stat__(row):
    doc = row["text"]
    row["n_tokens"] = len(doc.split())
    row["n_sents"] = len(list(nlp(doc).sents))
    return row


for dataset in datasets:
    if os.path.exists("corpus_stat/stat_{}_summary.txt".format(dataset)):
        continue

    print("Processing {}...".format(dataset))

    # download and unzip the dataset
    if "." in dataset:
        dataset_name, topic = dataset.split(".")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        data_path = os.path.join(data_path, topic)
    else:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)

    # get the distribution of corpus (number of tokens per document, number of sentences per document, number of documents)
    corpus = dataloader.read_corpus(data_path, num_proc=8)
    n_docs = len(corpus)
    n_samples = 10000
    if n_docs > n_samples:
        # randomly sample documents
        np.random.seed(42)
        random_indices = np.random.choice(n_docs, size=n_samples, replace=False)
        corpus = corpus.select(random_indices)
    else:
        n_samples = n_docs
    corpus = corpus.map(__stat__, batched=False, desc="Stat {}".format(dataset), num_proc=8)
    corpus_pd = corpus.to_pandas()

    corpus_summary = {}
    corpus_summary["n_docs"] = n_docs
    corpus_summary["n_docs_sampled"] = n_samples
    corpus_summary["n_tokens_sum"] = corpus_pd["n_tokens"].sum()
    corpus_summary["n_tokens_avg"] = corpus_pd["n_tokens"].mean()
    corpus_summary["n_tokens_median"] = corpus_pd["n_tokens"].median()
    corpus_summary["n_tokens_std"] = corpus_pd["n_tokens"].std()
    corpus_summary["n_tokens_lower_quartile"] = corpus_pd["n_tokens"].quantile(0.25)
    corpus_summary["n_tokens_upper_quartile"] = corpus_pd["n_tokens"].quantile(0.75)
    corpus_summary["n_sents_sum"] = corpus_pd["n_sents"].sum()
    corpus_summary["n_sents_avg"] = corpus_pd["n_sents"].mean()
    corpus_summary["n_sents_median"] = corpus_pd["n_sents"].median()
    corpus_summary["n_sents_std"] = corpus_pd["n_sents"].std()
    corpus_summary["n_sents_lower_quartile"] = corpus_pd["n_sents"].quantile(0.25)
    corpus_summary["n_sents_upper_quartile"] = corpus_pd["n_sents"].quantile(0.75)

    # save the corpus summary to text
    with open("corpus_stat/stat_{}_summary.txt".format(dataset), "w") as f:
        for k, v in corpus_summary.items():
            f.write("{}: {}\n".format(k, v))

    # plot the distribution of corpus
    (
        p9.ggplot(corpus_pd) + \
        p9.geom_histogram(p9.aes(x="n_tokens"), bins=100) + \
        p9.labels.ggtitle("{}: Number of Tokens per Document\n({} Samples from {} Docs, {} Tokens)".format(dataset, n_samples, n_docs, corpus_summary["n_tokens_sum"])) + \
        p9.labels.xlab("Number of Tokens") + \
        p9.labels.ylab("Number of Documents") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_tokens_avg"]), color="red") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_tokens_median"]), color="blue") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_tokens_lower_quartile"]), color="green") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_tokens_upper_quartile"]), color="green")
    ).save("corpus_stat/stat_{}_token.png".format(dataset), dpi=300)
    (
        p9.ggplot(corpus_pd) + \
            p9.geom_boxplot(p9.aes(x=0, y="n_tokens")) + \
            p9.labels.ggtitle("{}: Number of Tokens per Document\n({} Samples from {} Docs, {} Tokens)".format(dataset, n_samples, n_docs, corpus_summary["n_tokens_sum"])) + \
            p9.labels.xlab("") + \
            p9.labels.ylab("Number of Tokens") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_tokens_avg"]), color="red") + \
            p9.geom_text(p9.aes(x=0.3, y=corpus_summary["n_tokens_avg"]), label="AVG={:.2f}\nSTD={:.2f}".format(corpus_summary["n_tokens_avg"], corpus_summary["n_tokens_std"]), color="red") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_tokens_median"]), color="blue") + \
            p9.geom_text(p9.aes(x=0.1, y=corpus_summary["n_tokens_median"]), label="MED={:.0f}".format(corpus_summary["n_tokens_median"]), color="blue") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_tokens_lower_quartile"]), color="green") + \
            p9.geom_text(p9.aes(x=-0.1, y=corpus_summary["n_tokens_lower_quartile"]), label="LQ={:.0f}".format(corpus_summary["n_tokens_lower_quartile"]), color="green") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_tokens_upper_quartile"]), color="green") + \
            p9.geom_text(p9.aes(x=-0.2, y=corpus_summary["n_tokens_upper_quartile"]), label="UQ={:.0f}".format(corpus_summary["n_tokens_upper_quartile"]), color="green") + \
            p9.coord_flip()
    ).save("corpus_stat/stat_{}_token_box.png".format(dataset), dpi=300)
    (
        p9.ggplot(corpus_pd) + \
        p9.geom_histogram(p9.aes(x="n_sents"), bins=100) + \
        p9.labels.ggtitle("{}: Number of Sentences per Document\n({} Samples from {} Docs, {} Sentences)".format(dataset, n_samples, n_docs, corpus_summary["n_sents_sum"])) + \
        p9.labels.xlab("Number of Sentences") + \
        p9.labels.ylab("Number of Documents") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_sents_avg"]), color="red") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_sents_median"]), color="blue") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_sents_lower_quartile"]), color="green") + \
        p9.geom_vline(p9.aes(xintercept=corpus_summary["n_sents_upper_quartile"]), color="green")
    ).save("corpus_stat/stat_{}_sent.png".format(dataset), dpi=300)
    (
        p9.ggplot(corpus_pd) + \
            p9.geom_boxplot(p9.aes(x=0, y="n_sents")) + \
            p9.labels.ggtitle("{}: Number of Sentences per Document\n({} Samples from {} Docs, {} Sentences)".format(dataset, n_samples, n_docs, corpus_summary["n_sents_sum"])) + \
            p9.labels.xlab("") + \
            p9.labels.ylab("Number of Sentences") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_sents_avg"]), color="red") + \
            p9.geom_text(p9.aes(x=0.3, y=corpus_summary["n_sents_avg"]), label="AVG={:.2f}\nSTD={:.2f}".format(corpus_summary["n_sents_avg"], corpus_summary["n_sents_std"]), color="red") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_sents_median"]), color="blue") + \
            p9.geom_text(p9.aes(x=0.1, y=corpus_summary["n_sents_median"]), label="MED={:.0f}".format(corpus_summary["n_sents_median"]), color="blue") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_sents_lower_quartile"]), color="green") + \
            p9.geom_text(p9.aes(x=-0.1, y=corpus_summary["n_sents_lower_quartile"]), label="LQ={:.0f}".format(corpus_summary["n_sents_lower_quartile"]), color="green") + \
            p9.geom_hline(p9.aes(yintercept=corpus_summary["n_sents_upper_quartile"]), color="green") + \
            p9.geom_text(p9.aes(x=-0.2, y=corpus_summary["n_sents_upper_quartile"]), label="UQ={:.0f}".format(corpus_summary["n_sents_upper_quartile"]), color="green") + \
            p9.coord_flip()
    ).save("corpus_stat/stat_{}_sent_box.png".format(dataset), dpi=300)
