# Impact of embedders on Document Retrieval (Task 1) and Answer Generation (Task 3)
In addition to `dunzhang/stella_en_1.5B_v5`, we experimented on 2 other embedding models on the MTEB benchmark: `BAAI/bge-large-en-v1.5` and `all-mpnet-base-v2`. The three chunking strategies are denoted as: Fixed-size, Breakpoint, and Clustering. 

## Task 1: Document Retrieval
We present F1@k scores (%) from the 2 embedding models on Document Retrieval. 
A "stitched" dataset originally contained short documents, which we stitched into longer ones to enhance test set diversity.

**Observation**: Breakpoint-based chunking has a clear edge on datasets that originally contained short documents which were stitched together to form longer documents by us. In contrast, fixed-size chunking performs better on datasets with natively long documents. 

**Interpretation**: Short documents stitched by us are likely to be of unrelated topics. Hence, breakpoint-based chunking, which was given high hope by the LangChain and LlamaIndex communities, do not have an advantage for natively long documents. 

**Recommendation**: Just use fixed-size chunking in practice.

### Using `BAAI/bge-large-en-v1.5` as the embedder
**F1@1 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      59.81 |  **62.27** |      61.98 |
| NQ (stitched)           |      81.59 |  **85.62** |      81.60 |
| Scidocs (stitched)      |       6.37 |       7.62 |   **8.03** |
| Scifact (stitched)      |      44.02 |  **51.98** |      46.58 |
| BioASQ (stitched)       |      48.36 |  **52.56** |      45.83 |
| NFCorpus (stitched)     |      10.35 |      11.02 |  **11.43** |
| HotpotQA      |  **66.67** |  **66.67** |  **66.67** |
| MSMARCO       |  **95.00** |  **95.00** |  **95.00** |
| ConditionalQA |  **76.38** |      74.54 |      76.01 |
| Qasper        |  **92.45** |      90.84 |      90.84 |

**F1@2 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      70.10 |  **78.87** |      69.45 |
| NQ (stitched)           |      55.03 |  **66.53** |      52.36 |
| Scidocs (stitched)      |      14.13 |      16.39 |  **16.80** |
| Scifact (stitched)      |      37.98 |  **52.82** |      38.38 |
| BioASQ (stitched)       |      61.82 |  **62.26** |      61.53 |
| NFCorpus (stitched)     |      16.36 |      17.56 |  **18.53** |
| HotpotQA      |  **91.43** |      90.87 |      90.46 |
| MSMARCO       |      93.26 |  **93.56** |      92.64 |
| ConditionalQA |  **70.75** |      68.78 |      67.83 |
| Qasper        |  **89.18** |      87.62 |      88.65 |


**F1@5 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      63.78 |  **76.72** |      60.96 |
| NQ (stitched)           |      48.34 |  **57.05** |      45.95 |
| Scidocs (stitched)      |      18.06 |      18.47 |  **20.49** |
| Scifact (stitched)      |      29.24 |  **45.00** |      30.36 |
| BioASQ (stitched)       |      61.66 |  **66.36** |      61.39 |
| NFCorpus (stitched)     |      18.77 |  **19.79** |      19.75 |
| HotpotQA      |  **87.36** |      76.05 |      78.18 |
| MSMARCO       |  **90.93** |      90.09 |      89.51 |
| ConditionalQA |  **54.75** |      54.45 |      54.59 |
| Qasper        |  **81.03** |      76.33 |      77.15 |

**F1@10 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      45.19 |  **61.50** |      41.94 |
| NQ (stitched)           |      21.34 |  **30.81** |      20.19 |
| Scidocs (stitched)      |      17.19 |  **19.89** |      19.52 |
| Scifact (stitched)      |      19.57 |  **31.27** |      19.91 |
| BioASQ (stitched)       |      54.22 |  **62.65** |      53.35 |
| NFCorpus (stitched)     |      21.73 |      21.73 |  **22.27** |
| HotpotQA      |  **54.24** |      43.43 |      43.80 |
| MSMARCO       |  **78.90** |      76.65 |      68.90 |
| ConditionalQA |  **34.73** |      33.24 |      33.55 |
| Qasper        |  **58.67** |      46.86 |      42.97 |

### Using `SentenceBert/all-mpnet-base-v2` as the embedder
**F1@1 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      60.13 |  **62.97** |      61.08 |
| NQ (stitched)           |      81.53 |  **88.90** |      81.16 |
| Scidocs (stitched)      |   **5.40** |       5.00 |       5.33 |
| Scifact (stitched)      |      42.68 |  **50.70** |      45.36 |
| BioASQ (stitched)       |      40.68 |  **44.00** |      38.75 |
| NFCorpus (stitched)     |       7.21 |   **8.21** |       6.71 |
| HotpotQA      |  **62.67** |      62.00 |  **62.67** |
| MSMARCO       |  **98.00** |      94.00 |  **98.00** |
| ConditionalQA |  **72.69** |      71.96 |      71.59 |
| Qasper        |  **88.95** |      85.98 |      87.60 |

**F1@3 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      64.61 |  **73.95** |      67.21 |
| NQ (stitched)           |      53.30 |  **65.94** |      50.86 |
| Scidocs (stitched)      |      13.97 |      14.36 |  **16.42** |
| Scifact (stitched)      |      36.18 |  **51.07** |      36.59 |
| BioASQ (stitched)       |      49.54 |  **52.62** |      48.14 |
| NFCorpus (stitched)     |      12.39 |      13.55 |  **14.40** |
| HotpotQA      |  **78.69** |      76.53 |      75.40 |
| MSMARCO       |  **94.65** |      93.19 |      94.28 |
| ConditionalQA |  **72.63** |      69.00 |      68.69 |
| Qasper        |  **87.75** |      85.25 |      86.97 |

**F1@5 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      60.20 |  **69.93** |      59.72 |
| NQ (stitched)           |      37.05 |  **49.57** |      36.60 |
| Scidocs (stitched)      |      15.18 |      15.68 |  **17.85** |
| Scifact (stitched)      |      27.74 |  **43.76** |      28.99 |
| BioASQ (stitched)       |      47.42 |  **56.07** |      47.72 |
| NFCorpus (stitched)     |      14.71 |  **16.34** |      16.15 |
| HotpotQA      |  **71.73** |      65.09 |      65.84 |
| MSMARCO       |  **91.46** |      89.88 |      90.05 |
| ConditionalQA |  **60.46** |      56.10 |      60.44 |
| Qasper        |  **79.47** |      78.99 |      79.46 |

**F1@10 (%)**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| Miracl (stitched)       |      43.07 |  **51.27** |      41.25 |
| NQ (stitched)           |      21.49 |  **31.18** |      20.83 |
| Scidocs (stitched)      |      16.82 |      18.96 |  **19.50** |
| Scifact (stitched)      |      18.48 |  **29.28** |      18.78 |
| BioASQ (stitched)       |      42.58 |  **54.47** |      41.28 |
| NFCorpus (stitched)     |      18.16 |  **19.81** |      19.14 |
| HotpotQA      |  **47.21** |      40.56 |      40.06 |
| MSMARCO       |  **79.51** |      78.62 |      67.78 |
| ConditionalQA |  **38.04** |      36.63 |      36.44 |
| Qasper        |  **61.31** |      47.64 |      51.62 |

## Task 3: Answer Generation
We present BERTScore and QA Similarity scores from the 2 embedding models on Answer Generation.

**Takeaways**: Just use fixed-size chunking in practice. Breakpoint-based and chlustering-based have no advantage.

### Using `BAAI/bge-large-en-v1.5` as the embedder
**BERTScore**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| ExpertQA      |   **0.68** |       0.67 |       0.67 |
| DelucionQA    |       0.76 |   **0.78** |       0.76 |
| TechQA        |   **0.69** |       0.67 |       0.68 |
| ConditionalQA |   **0.40** |       0.39 |       0.39 |
| Qasper        |   **0.50** |       0.50 |       0.49 |

**QA Similarity**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| ExpertQA      |   **0.86** |       0.86 |       0.85 |
| DelucionQA    |       0.82 |       0.83 |   **0.83** |
| TechQA        |   **0.90** |       0.90 |       0.89 |
| ConditionalQA |       0.35 |   **0.36** |       0.36 |
| Qasper        |   **0.46** |       0.46 |       0.45 |

### Using `SentenceBert/all-mpnet-base-v2` as the embedder
**BERTScore**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| ExpertQA      |   **0.67** |       0.66 |       0.66 |
| DelucionQA    |       0.76 |   **0.76** |       0.76 |
| TechQA        |   **0.68** |       0.68 |       0.68 |
| ConditionalQA |       0.40 |       0.40 |   **0.41** |
| Qasper        |   **0.50** |       0.50 |       0.50 |

**QA Similarity**:

| Dataset       | Fixed-size | Breakpoint | Clustering |
| :------------ | ---------: | ---------: | ---------: |
| ExpertQA      |   **0.84** |       0.84 |       0.84 |
| DelucionQA    |       0.81 |       0.82 |   **0.83** |
| TechQA        |       0.86 |   **0.87** |       0.87 |
| ConditionalQA |       0.34 |   **0.35** |       0.34 |
| Qasper        |   **0.45** |       0.41 |       0.40 |
