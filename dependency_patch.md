# Patches

This file contains our patches to the dependencies of this project.

`.venv/lib/python3.10/site-packages/beir/retrieval/search/dense/exact_search.py:L73`:
```python3
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
```
change to
```python3
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[0])), dim=1, largest=True, sorted=return_sorted)
```
Reason: handle the case when there is only one document in the corpus
