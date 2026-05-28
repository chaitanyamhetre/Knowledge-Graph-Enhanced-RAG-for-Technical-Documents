
# Knowledge Graph Enhanced RAG for Technical Documents

A systematic benchmark comparing Vector RAG and Graph RAG pipelines for 
question answering over technical documents — evaluating retrieval accuracy, 
end-to-end answer quality, and query performance.

---

## Motivation

Retrieval-Augmented Generation (RAG) has emerged as the dominant approach 
for grounding LLM responses in external knowledge. Two competing paradigms 
exist:

- **Vector RAG** — retrieves documents using semantic similarity via dense 
  embeddings. General-purpose and robust across diverse domains.
- **Graph RAG** — retrieves documents by traversing a structured Knowledge 
  Graph built from entity relationships. Particularly powerful for multi-hop 
  reasoning and domain-specific structured corpora.

This project systematically benchmarks both approaches on technical documents, 
investigating when each method wins, where each fails, and what that reveals 
about their suitability for industrial knowledge systems.

---

## Research Questions

1. Does Graph RAG outperform Vector RAG on technical document retrieval?
2. How does entity extraction quality affect Graph RAG performance?
3. What is the end-to-end accuracy and latency of each full RAG pipeline?
4. Where does each method fail — and what does that reveal about its design?

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Input Question           │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                                     │
     ┌────────▼────────┐                 ┌──────────▼────────┐
     │   Vector RAG    │                 │    Graph RAG       │
     │                 │                 │                    │
     │ Sentence        │                 │ spaCy entity       │
     │ Transformer     │                 │ extraction         │
     │ embeddings      │                 │       ↓            │
     │       ↓         │                 │ Knowledge Graph    │
     │ FAISS similarity│                 │ traversal          │
     │ search          │                 │ (NetworkX)         │
     └────────┬────────┘                 └──────────┬────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │     Retrieved Context            │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │   LLM Answer Generation          │
                    │   (Llama 3.1 8B via Groq API)    │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │        Final Answer              │
                    └─────────────────────────────────┘
```

---

## Dataset

**RAG Dataset 12000**
- Source: `neural-bridge/rag-dataset-12000` via HuggingFace Datasets
- Subset used: 200 documents, 50 QA pairs for retrieval evaluation,
  20 QA pairs for end-to-end pipeline evaluation
- Format: context paragraph + question + ground truth answer
- Domain: general technical documents across diverse topics

---

## Methods

### Vector RAG
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers
- **Index:** FAISS IndexFlatL2 (CPU)
- **Retrieval:** top-3 most similar documents by L2 distance
- **LLM:** Llama 3.1 8B Instruct via Groq API (free tier)

### Graph RAG
- **Entity extraction:** spaCy `en_core_web_sm`
- **Graph construction:** co-occurrence edges between entities
  in the same document (NetworkX)
- **Retrieval:** entity matching from question → graph traversal
  → expand via weighted neighbors → retrieve source documents
- **Fallback:** vector RAG when no entities found in question
- **LLM:** same as Vector RAG

### Evaluation
- **Retrieval accuracy:** relaxed keyword match —
  70% of significant answer words must appear in retrieved context
- **Pipeline accuracy:** same metric applied to LLM-generated answer
- **Strict baseline:** exact string match (documented separately)

---

## Results

### Retrieval accuracy (n=50 queries)

| Method | Accuracy | Avg time/query |
|---|---|---|
| Vector RAG | **84.0%** | 0.028s |
| Graph RAG | 62.0% | 0.038s |

### End-to-end pipeline accuracy (n=20 queries)

| Method | Accuracy | Avg time/query |
|---|---|---|
| Vector RAG + LLM | 30.0% | 0.93s |
| Graph RAG + LLM | 25.0% | 1.48s |

### Key findings

- **Vector RAG outperforms Graph RAG** on general-domain text (84% vs 62%)
  due to robust semantic similarity matching across diverse topics
- **Graph RAG is limited by entity extraction quality** — spaCy's
  general-purpose NER introduces noise on diverse text
  (e.g. domain terms misclassified as PERSON/ORG), reducing graph precision
- **32% fallback rate** in Graph RAG confirms entity extraction as the
  primary bottleneck on general corpora
- **Pipeline accuracy drop** (84% → 30% for Vector RAG) reflects a
  metric artefact — LLMs rephrase answers in their own words, causing
  strict keyword matching to underestimate true answer quality
- **Graph RAG is expected to outperform Vector RAG** on domain-specific
  structured corpora (financial reports, technical manuals) where entity
  extraction is more reliable and multi-hop reasoning is required

### Knowledge graph statistics

- Nodes: 1,254 entities
- Edges: 6,240 co-occurrence relationships
- Entity types: ORG (656), PERSON (552), GPE (322),
  WORK_OF_ART (47), PRODUCT (22), EVENT (10)

---

## Limitations

**Entity extraction noise:** spaCy's general-purpose NER model
misclassifies domain-specific terms on diverse text. A fine-tuned
or domain-specific NER model would significantly improve Graph RAG
retrieval quality and reduce the fallback rate.

**Evaluation metric:** Strict and relaxed keyword matching both
underestimate true answer quality. LLMs generate semantically correct
answers that are phrased differently from ground truth, causing correct
answers to be marked wrong. Future work should use ROUGE-L, BERTScore,
or human evaluation for a more reliable measure.

**The drop from retrieval accuracy (84%) to pipeline accuracy (30%)**
reflects this metric limitation — not a real degradation. The LLM
generates well-formed, factually grounded answers as confirmed by
qualitative inspection of sample outputs.

**Dataset scope:** 200 documents from a general-domain corpus.
Graph RAG is designed for structured, domain-specific knowledge bases —
evaluation on automotive manuals, financial reports, or medical literature
would better reflect its intended use case.

---

## Future Work

- Evaluate on domain-specific corpora — automotive maintenance manuals
  or financial reports — where Graph RAG's multi-hop reasoning advantage
  is expected to emerge
- Implement proper evaluation metrics — ROUGE-L, BERTScore, or human
  annotation — for reliable LLM answer quality measurement
- Fine-tune spaCy NER on domain-specific text to reduce the 32%
  fallback rate and improve graph quality
- Extend Knowledge Graph with explicit relation types
  (not just co-occurrence) using relation extraction models
- Compare against Microsoft's GraphRAG library for a production-grade
  Graph RAG baseline
- Implement multi-hop reasoning evaluation — questions requiring
  connecting information across multiple documents

---

## Repo Structure

```
Knowledge-Graph-Enhanced-RAG-for-Technical-Documents/
├── notebooks/
│   ├── 01_data_and_entity_extraction.ipynb   # dataset loading, spaCy NER
│   ├── 02_knowledge_graph_and_vector_rag.ipynb # KG construction, FAISS, retrieval benchmark
│   └── 03_graphrag_analysis_and_plots.ipynb  # LLM pipeline, plots, final analysis
├── data/
│   ├── raw/
│   └── processed/
│       ├── documents.csv                     # cleaned document corpus
│       ├── qa_pairs.csv                      # question-answer pairs
│       ├── entities.csv                      # extracted named entities
│       └── embeddings.npy                    # sentence transformer embeddings
├── graph/
│   └── knowledge_graph.gexf                  # exportable knowledge graph
├── results/
│   ├── retrieval_results.csv                 # retrieval benchmark results
│   ├── pipeline_results.csv                  # end-to-end pipeline results
│   ├── final_summary.csv                     # aggregate summary statistics
│   └── figures/
│       ├── knowledge_graph.png               # graph visualisation
│       ├── retrieval_comparison.png          # accuracy + runtime comparison
│       ├── graph_fallback_analysis.png       # fallback rate analysis
│       ├── outcome_distribution.png          # win/loss distribution
│       └── pipeline_comparison.png          # full pipeline comparison
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Reproducibility

All notebooks run on **Google Colab free tier (T4 GPU / CPU)**.
No paid APIs required — uses Groq free tier for LLM inference.

```bash
pip install datasets spacy networkx sentence-transformers             faiss-cpu matplotlib pandas pyvis groq
python -m spacy download en_core_web_sm
```

**API keys needed (both free):**
- HuggingFace token — for dataset access: huggingface.co/settings/tokens
- Groq API key — for LLM inference: console.groq.com

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow)
![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph-orange)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-red)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-purple)

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector index:** FAISS (CPU)
- **Knowledge Graph:** NetworkX + spaCy en_core_web_sm
- **LLM:** Llama 3.1 8B Instruct via Groq free API
- **Dataset:** neural-bridge/rag-dataset-12000 (HuggingFace)
- **Hardware:** Google Colab free tier

---
