# Transformer-based Neural Machine Translation (German to English)

## Overview

This project implements a **Neural Machine Translation (NMT)** system based on the **Transformer architecture**. The model is trained to translate **German sentences** into **English**. Transformer models have become the backbone of state-of-the-art language models due to their ability to handle long-range dependencies in sequences using the **self-attention mechanism**.

The project includes:
- **Model architecture**: Transformer Encoder-Decoder architecture
- **Training**: Training on a parallel German-English corpus
- **Evaluation**: Using BLEU score to measure translation accuracy

## Architecture

The model consists of two main components:

1. **Encoder**: Processes the input sequence (German) using multi-head self-attention to generate embeddings.
2. **Decoder**: Takes the encoded German embeddings and generates the English translation, also using multi-head attention.
3. **Positional Encoding**: Since transformers don’t inherently capture the order of tokens, positional encoding is added to the input embeddings to provide information about the position of words.
4. **Feed-forward Networks**: Both encoder and decoder have a feed-forward network for further transformation of the embeddings.
5. **Residual Connections and Layer Normalization**: Applied to stabilize training and speed up convergence.

## Requirements

The following libraries are required to run this project:

- **TensorFlow**: A deep learning framework used to build and train the model.
- **NumPy**: For handling arrays and numerical operations.
- **NLTK**: For natural language processing tasks such as calculating the BLEU score.
- **Pandas**: For handling and preprocessing the dataset.
- **Matplotlib**: (Optional) For visualizations.

To install these dependencies, create a **virtual environment** and install the required libraries by running:

```bash
pip install -r requirements.txt
```

## SETUP
### 1. Clone the Repository
```bash
git clone https://github.com/DritiSanaja/transformer-nmt.git
cd transformer-nmt

```

### 2. Create and Activate a Virtual Environment
Create a `.env` file in the root directory or set environment variables to store your API key:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset
This project uses the Europarl dataset for training. You can download it from the Europarl website or use the preprocessed version.


- **europarl-v7.de-en.de: German text file**
- **europarl-v7.de-en.en: English text file**

## Training the Model

### 1. Preprocess Data

Before training, the dataset needs to be preprocessed:

- Tokenizing the text data (converting words into numerical tokens)
- Padding the sequences to ensure they are of uniform length
- Adding special tokens (`< SOS >`, `<EOS>`) to mark the start and end of each sentence

### 2. Train the Model

You can start training the model using the following code:

```bash
tran.fit((preproc_german_sentences, preproc_english_sentences[:, :-1]),
         preproc_english_sentences[:, 1:, tf.newaxis],
         epochs=10, verbose=True, batch_size=64)
```

- **preproc_german_sentences: The tokenized and padded German input sentences.**
- **preproc_english_sentences: The tokenized and padded English target sentences (with the EOS token removed from the input for prediction).**
- **The model will train for 10 epochs, but you can adjust the number of epochs depending on your dataset size and available resources.**

### 3. Save the Model

```bash
tran.save("nmt_model.h5")

```


## Evaluation

After training, you can evaluate the model using the BLEU score, which measures the overlap between predicted n-grams and the true translations. Here's how to evaluate the model:

```bash
from nltk.translate.bleu_score import sentence_bleu

def show():
    i = random.randint(0,170111)

    print("German sentence : ", german_tokenizer.sequences_to_texts(preproc_german_sentences[[i]]))

    predict_sent = pred(i)[0]
    print("Predicted sentence : ", predict_sent)

    true_sent = english_tokenizer.sequences_to_texts(preproc_english_sentences[[i]])[0]
    print("True sentence : ", true_sent)

    predict_sent_words = predict_sent.split(' ')
    true_sent_words = true_sent.split(' ')

    if predict_sent_words[0] == '<SOS>':
        predict_sent_words = predict_sent_words[1:]

    bleu_score = sentence_bleu([true_sent_words], predict_sent_words)
    print('BLEU score: {}'.format(bleu_score))
```
## Results

After training for one epoch, the model starts generating translations with relatively low BLEU scores (e.g., 0.12). This is expected since the model is still in its early training stages.
For improved results, you should train the model for more epochs (e.g., 10–50 epochs). Increasing the model complexity (more layers, larger embedding sizes) or fine-tuning the hyperparameters may also help.

## Challenges and Future Improvements

### 1. **Insufficient Training**:
The model was trained for just one epoch, and the BLEU scores are low. Increasing the training epochs should improve performance. It's common for transformer models to require extensive training to converge.

### 2. **Model Hyperparameters**:
* Experiment with different **embedding sizes**, **number of layers**, and **number of attention heads**.
* Fine-tune the learning rate to see how it affects the training process.

### 3. **Evaluation Metrics**:
BLEU is a commonly used metric for evaluating machine translation quality, but other metrics such as **METEOR** and **TER** may provide a better understanding of translation quality.

### 4. **Preprocessing and Data Augmentation**:
* Proper tokenization and handling of punctuation are crucial for achieving good translation quality.
* Consider augmenting the dataset or fine-tuning on domain-specific data for more accurate translations in specific contexts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Conclusion

This repository provides a foundational **Transformer-based Neural Machine Translation** model for translating German to English. While the model is still in its early stages, it offers a great starting point for exploring **Transformer architectures** in machine translation tasks. With more training and experimentation with hyperparameters, the model's performance can be significantly improved.

Feel free to fork, clone, and improve the code for your own translation tasks!
