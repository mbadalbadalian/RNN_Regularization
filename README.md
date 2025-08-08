# Recurrent Neural Network Regularization

_Replicating and Extending Dropout Techniques in LSTMs and GRUs Across NLP Tasks_

This project reimplements and evaluates the core findings of Zaremba et al. (2014) by exploring dropout applied only to non-recurrent connections in RNNs. The experiments span three key sequence learning tasks: language modeling, machine translation, and image captioning, with custom LSTM and GRU architectures built entirely from scratch in PyTorch.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Task & Dataset Descriptions](#-task--dataset-descriptions)
- [How It Works](#-how-it-works)
- [Results](#-results)
- [Running the Code](#️-running-the-code)
- [Authors](#️-authors)
- [License](#-license)

---

## Project Overview

We investigate how selective dropout, applied only to non-recurrent connections, impacts the performance of recurrent neural networks. This is done for three seperate tasks: language modeling, machine translation and image caption generation. Our implementation language modeling is inspired by the architecture and training regime from the original paper while testing the generalization of this technique across multiple datasets and RNN types (LSTM, GRU).

We further perform an extensive hyperparameter search, build a flexible training framework, and generate comparative visualizations of training/validation perplexity.

---

## Features

### 1. Custom RNN Implementations
- Fully custom `LSTMCell`, `GRUCell`, and corresponding models
- Dropout applied *only* to input and inter-layer connections

### 2. Multi-Task Pipeline
- Language Modeling (Tiny Shakespeare)
- Machine Translation (EN–FR, Tatoeba)
- Image Captioning (Flickr8k)

### 3. Experimental Framework
- Model checkpoints + early stopping
- BLEU and Perplexity evaluation metrics
- Hyperparameter tuning over 70+ experiments

### 4. Visual Analysis
- Epoch-wise perplexity plots
- Model comparison tables
- Train vs. Validation tracking

---

## Tech Stack

| Tool / Library     | Purpose                              |
|--------------------|---------------------------------------|
| `PyTorch`          | Deep learning framework               |
| `datasets`         | Access to Tiny Shakespeare & Tatoeba  |
| `sklearn`          | Train/validation split                |
| `matplotlib`       | Metric visualization                  |
| `tqdm`             | Progress bars                         |
| `pickle`, `os`     | Model/metric I/O                      |
| `nltk`             | BLEU score calculation                |

---

## Task & Dataset Descriptions

| Task               | Dataset        | Description                                                  |
|--------------------|----------------|--------------------------------------------------------------|
| **Language Modeling**   | Tiny Shakespeare | Small corpus for next-word prediction tasks                |
| **Machine Translation** | Tatoeba (EN–FR)  | Parallel EN–FR corpus with ~500k sentence pairs            |
| **Image Captioning**    | Flickr8k         | 8,000 images with 5 human-written captions each            |

All datasets are accessed via Hugging Face Datasets or standard open-access repositories.

---

## How It Works

### Architecture

- Stacked LSTM and GRU models with flexible dropout
- Sequence-to-sequence pipeline for translation and captioning
- Beam search & greedy decoding for inference

### Training Loop

- Gradient clipping
- Learning rate decay
- Early stopping based on validation metric

### Evaluation

- Perplexity for Language Modeling
- BLEU for MT & Captioning
- Metric tracking across all epochs and model versions

---

## Results

### Language Modeling (Tiny Shakespeare)

| Model                 | Train PPL | Valid PPL | Test PPL |
|----------------------|-----------|-----------|----------|
| LSTM (no dropout)     | 437.8     | 318.2     | 318.5    |
| LSTM (dropout)        | 436.4     | 319.1     | 327.6    |
| GRU (no dropout)      | 440.6     | 286.3     | 314.7    |
| GRU (dropout)         | 455.1     | 301.3     | 324.8    |

> Dropout provided minimal benefit in LM under limited training budget.

### Machine Translation (Tatoeba EN–FR)

| Model                 | Train PPL | Valid PPL | Test PPL |
|----------------------|-----------|-----------|----------|
| LSTM (no dropout)     | 2.11      | 2.98      | 2.97     |
| LSTM (dropout)        | 2.10      | 2.95      | 2.92     |
| GRU (no dropout)      | 1.97      | 3.10      | 3.03     |
| GRU (dropout)         | 1.50      | 2.80      | 2.70     |

> Dropout improves generalization, particularly in GRUs.

### Image Captioning (Flickr8k)

| Model                 | Train PPL | Valid PPL | Test PPL |
|----------------------|-----------|-----------|----------|
| LSTM (no dropout)     | 3.05      | 4.69      | 4.60     |
| LSTM (dropout)        | 3.07      | 4.61      | 4.51     |
| GRU (no dropout)      | 3.28      | 5.02      | 4.87     |
| GRU (dropout)         | 3.20      | 4.50      | 4.42     |

> Dropout improves generalization in the case of GRUs.

### Key Findings

- **Selective Dropout**: Applying dropout only to non-recurrent connections improves generalization in LSTMs and GRUs, especially when trained on larger datasets with sufficient time.
- **GRU Performance**: GRUs show significant improvements with selective dropout, particularly in sequence generation tasks, compared to LSTMs.
- **Training Time & Data**: The benefits of dropout are most noticeable with longer training times and larger datasets, which were constrained in some experiments.
- **Internal Recurrence**: Dropout must be applied carefully to avoid disrupting the crucial recurrent pathways in RNNs, which are key for long-term dependencies.

---

## Running the Code

### Setup

Clone the repo and install dependencies:
```bash
pip install torch datasets numpy matplotlib nltk scikit-learn tqdm
```
or:
```bash
pip install -r requirements.txt
```

### Execute Main Scripts

Each notebook can be ran interactively or programmatically:

For language modeling:
```bash
jupyter notebook notebooks/language_modeling.ipynb
```
or:
```bash
jupyter nbconvert --to notebook --execute notebooks/language_modeling.ipynb --output output_language_modeling.ipynb
```

For machine caption generation:
```bash
jupyter notebook notebooks/machine_translation.ipynb
```
or:
```bash
jupyter nbconvert --to notebook --execute notebooks/machine_translation.ipynb --output output_machine_translation.ipynb
```

For image caption generation:
```bash
jupyter notebook notebooks/image_captioning.ipynb
```
or:
```bash
jupyter nbconvert --to notebook --execute notebooks/image_captioning.ipynb --output output_image_captioning.ipynb
```

---

## Authors

**Matthew Badal-Badalian**  
*MDSAI Graduate, University of Waterloo*  
- [LinkedIn](https://www.linkedin.com/in/badal/)    
- [GitHub](https://github.com/mbadalbadalian)

**Aman Sharma**  
*MDSAI Graduate, University of Waterloo*  
- [LinkedIn](https://www.linkedin.com/in/aman26sharma/)    
- [GitHub](https://github.com/Aman26Sharma)

---

## License
This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

Portions of this work are adapted from [wojzaremba/lstm](https://github.com/wojzaremba/lstm), originally written in Lua by Wojciech Zaremba and collaborators.

You can find the full license text in the [`LICENSE`](./LICENSE) file included in this repository.
