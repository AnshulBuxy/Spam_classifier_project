
# Spam Classifier

It is a machine learning-based **Spam Classifier** that predicts whether a given message is spam or not. The classifier uses natural language processing (NLP) techniques for text preprocessing, vectorization, and classification. This README explains the workflow, steps involved, and how to use the project.

---

## Workflow

1. **Dataset Preparation**
2. **Exploratory Data Analysis (EDA)**
3. **Text Preprocessing and Vectorization**
4. **Model Building and Training**
5. **Model Evaluation**

---

## Steps Explained

### 1. Dataset Preparation
- The dataset used consists of text messages labeled as either `spam` or `ham` (not spam).
- Missing values are handled, and the text is preprocessed to remove unnecessary characters, punctuations, and stopwords.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of messages (spam vs. non-spam).
- Visualized common words in both spam and non-spam messages using word clouds.

![EDA Results](eda_results.png)  

### 3. Text Preprocessing and Vectorization
- The text messages are tokenized and converted into numerical representations using:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - Bag-of-Words for feature extraction.

### 4. Model Building and Training
- Split the dataset into training and testing sets (e.g., 80-20 split).
- Used classification algorithms such as:
  - **Naive Bayes**
  - **Logistic Regression**
- Trained the models on vectorized text data.

### 5. Model Evaluation
- Evaluated the model using the following metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- A confusion matrix was generated to visualize the performance.

---

## Results
- The classifier achieved high accuracy on the test set, demonstrating effective spam detection.
- Key metrics:
  - **Accuracy:** 95%
  - **Precision:** 96%
  - **Recall:** 93%
  - **F1-Score:** 94%

---

## How to Use

1. **Clone the Repository**
   ```bash
   git clone <[repository-link](https://github.com/AnshulBuxy/Spam_classifier_project)>
   cd spam-classifier
   ```

3. **Run the Notebook**
   - Open the `spamclassifier.ipynb` file in Jupyter Notebook or Google Colab.
   - Follow the steps to train the model and evaluate it.

4. **Make Predictions**
   - Use the provided function to input a text message and predict whether it is spam or not.


---

## Future Improvements
- Integrate the classifier into a web application using **Streamlit** or **Flask**.
- Enhance preprocessing to handle multilingual messages.
- Experiment with advanced NLP models like **BERT** for improved performance.

---

