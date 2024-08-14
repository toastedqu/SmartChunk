import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
import seaborn as sns

metrics = ["NDCG@k", "MAP@k", "Recall@k", "Precision@k", "MRR@k", "R_cap@k", "Hole@k", "Accuracy@k"]
plot_metrics = ['P@k' if metric == 'Precision@k' else metric for metric in metrics] + ["Latency"]
cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
beir_datasets = ['nfcorpus', 'nq', 'hotpotqa', 'fiqa', 'webis-touche2020', 'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact']
longdoc_datasets = ['mldr']
k_values = [1,3,5,10,20]

def get_important_metrics(metrics: list):
    important_metrics = []
    for metric in metrics:
        if metric == "Precision@k":
            for k in k_values:
                important_metrics.append(f'P@{str(k)}')
        else:
            for k in k_values:
                important_metrics.append(metric.replace('k',str(k)))
    important_metrics.append('Latency')
    return important_metrics

important_metrics = get_important_metrics(metrics)

def get_results_df(dataset: str):
    results = []
    with open(f"results/results.jsonl",'r') as f:
        for line in f:
            result = json.loads(line)
            if result['dataset'] == dataset:
                results.append(result)

    df = pd.DataFrame(results)
    for metric in metrics:
        df = pd.concat([df, df[metric].apply(pd.Series)], axis=1)
        df.drop(metric, axis=1, inplace=True)

    return df.loc[:, ['dataset','encoder','chunker','hyperparams'] + important_metrics]

def get_average_results(datasets=None):
    datasets = cfg['datasets'] if datasets is None else datasets
    dfs = []
    for dataset in datasets:
        df = get_results_df(dataset).drop('dataset', axis=1)
        df['hyperparams'] = df['hyperparams'].astype(str)
        dfs.append(df)
    df = pd.concat(dfs).groupby(by=['encoder', 'chunker', 'hyperparams']).mean().reset_index()
    df['hyperparams'] = df['hyperparams'].apply(lambda x: eval(x))
    return df.sort_values(important_metrics, ascending=[False]*(len(important_metrics)-1)+[True])

def get_best_config_per_chunker(df, chunker, metric_k):
    df_sub = df.loc[df['chunker']==chunker, metric_k]
    return df_sub.index[df_sub.argmax()]

def get_best_configs(df, metric):
    idxs = [get_best_config_per_chunker(df, chunker, metric) for chunker in df['chunker'].unique()]
    return df.loc[idxs, ['chunker', metric]]

def get_method_comparison_for_metric(metric, df=None):
    if df is None: df = get_average_results()
    temp_df = pd.DataFrame({"chunker": df['chunker'].unique()})
    if metric == 'Latency':
        temp_df = pd.merge(temp_df, get_best_configs(df, metric), on='chunker')
    else:
        for k in k_values:
            temp_df = pd.merge(temp_df, get_best_configs(df, metric.replace('k',str(k))), on='chunker')
    return temp_df.melt(id_vars=['chunker'], var_name=metric, value_name='score')

def plot_results_for_best_configs_per_method(dataset: Optional[str] = None):
    if dataset is None:
        df = get_average_results()
    elif dataset == 'beir':
        df = get_average_results(beir_datasets)
    elif dataset == 'longdoc':
        df = get_average_results(longdoc_datasets)
    else:
        df = get_results_df(dataset)
    dataset_title = dataset if dataset else 'all datasets'
    fig, axes = plt.subplots(len(plot_metrics), 1, figsize=(10, len(plot_metrics)*3))
    for i,metric in enumerate(plot_metrics):
        ax = axes[i]
        if i == 0: ax.set_title(f"Comparison of methods with best hyperparameters for {dataset_title}")
        temp_df = get_method_comparison_for_metric(metric, df)
        metric_name = metric.split('@')[0]
        sns.barplot(x=metric, y='score', hue='chunker', data=temp_df, ax=ax, zorder=3)
        ax.set_xlabel('')
        ax.set_ylabel(f'{metric_name} score')
        delta = 5 if metric == 'Latency' else 0.05
        ax.set_ylim(max(temp_df['score'].min()-delta, 0), temp_df['score'].max()+delta)
        ax.legend()
        ax.grid(axis='y', zorder=0)
    fig.tight_layout()
    plt.show()

def get_hyperparam_comparison(df, chunker):
    df_sub = df[df['chunker']==chunker].reset_index(drop=True)
    return pd.concat([df_sub.drop('hyperparams', axis=1), pd.json_normalize(df_sub['hyperparams'])], axis=1)
