# Comparative Analysis of SVM-BERT and CNN-BERT in Relation Extraction

This repository contains the project for the Text Mining elective, which implements and compares two different models for relation extraction on the FewRel dataset.

## 👥 Group Members

* Mohammad Wathiq Soualhi - w51088ms
* Mueez Nadeem - j62745mn
* Taha Mulla - b50695tm

## 📜 Abstract

This project presents an implementation of two models for the Few-Shot Relation Classification Dataset (FewRel): a **Support Vector Machine (SVM)** using BERT embeddings and a **DistilBERT-Convolutional Neural Network (CNN)** approach. The models are evaluated based on Precision, Recall, F1 Score, and Accuracy.

---

## 📖 Introduction

Relation Extraction (RE) is a key task in Natural Language Processing (NLP) that aims to identify and classify semantic relationships between entities in text. This study uses the **FewRel dataset**, a benchmark for few-shot relation extraction, to compare a traditional machine learning approach (SVM) with a deep learning method (CNN-DistilBERT) for practical RE applications.

---

## 💾 Dataset

The project uses the **FewRel dataset**, a large-scale supervised few-shot relation classification dataset. The notebooks source data from `train_wiki.json` (training set) and `val_wiki.json` (validation set). For the SVM model, these are concatenated to create a dataset of 11,200 instances for supervised learning.

---

## ⚙️ Methodologies

This project implements and compares two different approaches to relation extraction.

### 1. Support Vector Machine (SVM) with Hybrid Features

This model uses a Support Vector Machine (SVM) classifier, which is effective in high-dimensional feature spaces. To enhance its performance, a hybrid feature set is used, combining linguistic features with modern embeddings.

* **Feature Extraction**: A `ColumnTransformer` is used to create a 975-dimensional feature vector per instance by combining:
    * **Linguistic Features**: NER labels, POS tags, dependency relations, and token distance between entities, extracted using **spaCy**.
    * **GloVe Embeddings**: 200-dimensional semantic representations from pre-trained 100D GloVe embeddings (`glove-wiki-gigaword-100`), averaged and concatenated for the head and tail entities.
    * **BERT Embeddings**: 768-dimensional sentence-level embeddings generated using a custom `BertEmbedder` with `bert-base-uncased`.
* **Model Training**: A scikit-learn `Pipeline` is used to chain the feature extraction, scaling (`MaxAbsScaler`), and classification steps. The classifier is an `SVC` with an RBF kernel.

### 2. CNN-DistilBERT

This model is a hybrid deep learning approach that combines a Convolutional Neural Network (CNN) with DistilBERT to balance performance and computational efficiency.

* **Architecture**:
    * **DistilBERT Embedding Layer**: A pre-trained `distilbert-base-uncased` model generates contextualized word embeddings.
    * **Convolutional Layers**: 1D convolutional layers are used to extract local patterns from the sequence of embeddings, followed by ReLU activation.
    * **Max Pooling and Dropout**: These layers are used to reduce dimensionality and prevent overfitting.
    * **Fully Connected Layer**: A final dense layer maps the features to the number of relation classes for classification.
* **Model Training**: The model is trained using cross-entropy loss and the AdamW optimizer. The notebook specifies training for 3 epochs with a batch size of 8 and a learning rate of 2e-5.

---

## 📊 Evaluation & Results

The two models were evaluated on the test set, and their performance was compared.

* **SVM-BERT**: This model achieved a strong macro-averaged **F1-score of 0.78**. It performed very well on relations with clear linguistic patterns (e.g., P412 with a 1.00 F1-score) but struggled with more ambiguous relations (e.g., P40 with a 0.19 F1-score).
* **CNN-DistilBERT**: This model performed poorly, with a very low **accuracy of 0.79%** and a weighted **F1-score of 1.21%**.

**Conclusion**: The SVM-BERT hybrid approach significantly outperformed the CNN-DistilBERT model, demonstrating that in low-resource or few-shot scenarios, a well-designed feature-based method can be more effective than a deep learning model that has not been extensively tuned.

---

## 🚀 How to Use

### Notebooks

* **`cnn-distillbert_final.ipynb`**: This notebook implements the CNN-DistilBERT model for relation extraction. It uses pre-trained DistilBERT embeddings and a CNN architecture to classify relationships.
* **`SVM.ipynb`**: This notebook implements the Support Vector Machine (SVM) model for relation extraction, providing a simpler baseline for comparison.

### Running the Code

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies**: The notebooks require `transformers`, `torch`, `datasets`, `scikit-learn`, `spacy`, and `gensim`.
    ```bash
    pip install transformers torch datasets scikit-learn spacy gensim
    python -m spacy download en_core_web_sm
    ```

3.  **Run the Notebooks**: Open and run the Jupyter notebooks.
    * The `SVM.ipynb` notebook will clone the FewRel repository to access the dataset. If you run the training part first, you do not need to clone it again for the inference mode.
    * For the SVM inference, ensure the `svm_re_pipeline.pkl` model file is placed inside the `FewRel` directory.

---
