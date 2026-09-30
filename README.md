# BERT Sentiment Fine-Tuning on IMDb Reviews

A natural language processing project that fine-tunes a pretrained BERT model to classify movie reviews as **positive** or **negative**.

This repository demonstrates the complete workflow from exploring and preparing text data to training, evaluating, and making predictions with a transformer model. The notebook's saved run achieved **86.44% test accuracy on 450 held-out reviews**, using a balanced sample of 3,000 reviews for the experiment.

## Purpose

Large collections of written feedback are difficult to review manually. Sentiment classification helps organize that feedback by identifying whether a piece of text expresses a positive or negative opinion.

This project uses movie reviews as a practical example. Similar techniques can support customer feedback analysis and review monitoring, although applying this model to another domain would require additional data and evaluation.

Rather than training a language model from scratch, the project adapts `bert-base-uncased` to a specific classification task. This approach is called **fine-tuning**: the pretrained model's weights are updated using labeled examples of movie reviews.

## Technologies Used

| Technology | Role in the project |
|---|---|
| Python | Data preparation, training, evaluation, and prediction |
| Hugging Face Transformers | Pretrained BERT tokenizer and sequence classification model |
| PyTorch | Custom Dataset, DataLoader batching, training loop, loss calculation, and GPU support |
| Pandas and NumPy | Data handling and exploration |
| Scikit-learn | Splitting reviews into training, validation, and test sets |
| Matplotlib and Seaborn | Visualizing sentiment distribution, review lengths, and training metrics |
| Python regular expressions | Cleaning HTML tags, punctuation, and extra whitespace |
| Jupyter Notebook / Google Colab | Running and documenting the experiment interactively |

## Dataset

The input file, `IMDB Dataset.csv`, contains 50,000 movie reviews with two columns:

- `review`: the movie review text.
- `sentiment`: the positive or negative label.

For this experiment, the notebook samples **1,500 positive and 1,500 negative reviews**, then shuffles them using `random_state=42`.

| Split | Reviews | Purpose |
|---|---:|---|
| Training | 2,100 (70%) | Update model weights |
| Validation | 450 (15%) | Monitor performance after each training epoch |
| Test | 450 (15%) | Evaluate the trained model on held-out reviews |

The splits use a fixed random state but do not explicitly stratify by label. The experiment uses the CSV-based split described above, rather than the original IMDb benchmark train/test partition.

## Workflow

1. **Explore the data:** inspect review counts, sentiment balance, missing values, and review lengths.
2. **Prepare the text:** remove HTML tags, punctuation, and extra whitespace; convert text to lowercase. Encode negative as `0` and positive as `1`.
3. **Split the data:** reserve separate training, validation, and test sets.
4. **Tokenize reviews:** use the BERT tokenizer to convert text into model inputs, padding or truncating each review to 128 tokens.
5. **Create batches:** wrap tokenized reviews and labels in a custom PyTorch Dataset and DataLoader.
6. **Fine-tune BERT:** train a two-class BERT model for three epochs using AdamW and cross-entropy loss. Use CUDA when available, with a CPU fallback.
7. **Monitor learning:** record training loss, validation loss, and validation accuracy; visualize changes across epochs.
8. **Evaluate and predict:** calculate test loss and accuracy, then classify example reviews and return predicted labels with model probabilities.

## Training Configuration

| Setting | Value |
|---|---|
| Pretrained model | `bert-base-uncased` |
| Output classes | Negative and positive |
| Maximum input length | 128 tokens |
| Batch size | 64 |
| Epochs | 3 |
| Optimizer | PyTorch AdamW |
| Learning rate | `1e-5` |
| Optimizer epsilon | `1e-8` |
| Loss function | CrossEntropyLoss |
| Compute device | CUDA if available; otherwise CPU |

## Recorded Results

The uploaded notebook contains the following saved results:

| Metric | Value |
|---|---:|
| Final validation accuracy | 85.11% |
| Test accuracy | 86.44% |
| Test loss | 0.3216 |

These values describe one recorded run; they have not been independently rerun for this README. Training randomness and hardware differences can affect subsequent results. Accuracy alone does not describe all classification errors; precision, recall, F1-score, and a confusion matrix would provide a fuller evaluation.

## Repository Files

| File | Description |
|---|---|
| `bert_sentiment_finetuning.ipynb` | Data exploration, preprocessing, model training, evaluation, and sample inference |
| `IMDB Dataset.csv` | Review text and sentiment labels used by the notebook |
| `README.md` | Project overview and instructions |

## How to Run

### In VS Code or Jupyter

Install the notebook's dependencies in your Python environment:

```bash
python -m pip install torch transformers pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel
```

Open `bert_sentiment_finetuning.ipynb` and select that Python environment as the notebook kernel. The notebook currently loads the CSV from a Google Colab path:

```python
data = pd.read_csv("/content/IMDB Dataset.csv")
```

For local use, replace it with the following line and run the notebook with the repository folder as the working directory:

```python
data = pd.read_csv("IMDB Dataset.csv")
```

Run the cells in order. Internet access is needed for the initial download of the pretrained tokenizer and model. A GPU is recommended for training; reduce the batch size if available GPU memory is insufficient. The dependencies are not version-pinned in this repository.

### In Google Colab

Upload the notebook and dataset, select a GPU runtime if available, and place the CSV at `/content/IMDB Dataset.csv`. Run the cells in order.

## Sample Predictions

After training, the notebook's `predict_sentiment()` function accepts a single review or a list of reviews:

```python
reviews = [
    "This movie was absolutely brilliant — great acting and a touching story.",
    "Terrible waste of time. Boring plot and wooden performances.",
]

results = predict_sentiment(reviews, model, tokenizer)
for result in results:
    print(result)
```

Each result includes a text preview, predicted sentiment, the predicted class probability, and probabilities for both classes. These probabilities are model scores, not calibrated guarantees of correctness. This example uses the model trained in the current notebook session.

## Scope and Next Steps

This repository is a notebook-based training experiment. Long reviews are truncated to 128 tokens, and the training sample is smaller than the full dataset. The inference helper currently tokenizes raw input without applying the training text-cleaning function.

Useful extensions include consistent preprocessing during inference, stratified splits, full training seed control, a baseline comparison, richer evaluation metrics, and saving the trained model and tokenizer for reuse. A prediction API or web interface could follow after those improvements.

## Author

**Ritika Dharamkar**

- [GitHub](https://github.com/RitikaDharamkarJ)
- [LinkedIn](https://www.linkedin.com/in/ritikadharamkar/)
