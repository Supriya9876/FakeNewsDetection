# Fake News Detection using Machine Learning and Deep Learning

<h2> Description</h2>

Fake news has become a significant issue in today's digital age, affecting public opinion and spreading misinformation rapidly. The proliferation of social media platforms and online news sources has made it easier for false information to be disseminated widely. This project aims to tackle the problem of fake news by using machine learning techniques to automatically detect and classify news articles as either fake or real.

<h2>Project Workflow</h2>

The project workflow is outlined as follows:

1. **Data Collection:** Acquire a dataset of labeled news articles.
2. **Data Preprocessing:** Clean and preprocess the text data to remove noise and standardize formats.
3. **Feature Extraction:** Transform the text data into numerical features that can be fed into machine learning models.
4. **Model Training:** Train multiple machine learning models using the processed data.
5. **Model Evaluation:** Assess the performance of each model and select the best-performing one.
6. **Deployment:** Develop an interface or application for practical use.

<h2>Model Selection</h2>

For the fake news detection project, we have chosen to use two distinct models: Logistic Regression and BERT (Bidirectional Encoder Representations from Transformers). These models represent a combination of traditional machine learning and advanced deep learning techniques, providing a robust approach to classifying news articles as fake or real.

### 1. Logistic Regression

**Overview**:
Logistic Regression is a widely used statistical model for binary classification problems. It is simple yet effective and often serves as a baseline model for text classification tasks.

**Steps**:

- **Feature Extraction**: We utilize TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert the text data into numerical features. TF-IDF captures the importance of each word in a document relative to the entire dataset.
- **Model Training**: The Logistic Regression model is trained on the TF-IDF features of the news articles. The training process involves fitting the model to the data and optimizing the parameters to minimize the classification error.
- **Performance Evaluation**: The trained Logistic Regression model is evaluated using metrics such as accuracy, precision, recall, and F1 score. These metrics help assess the model's ability to correctly classify news articles.

**Advantages**:

- Simple and interpretable.
- Fast training and inference.
- Effective for linearly separable data.

**Limitations**:

- May not capture complex patterns in the data.
- Less effective for highly non-linear relationships.

### 2. BERT Model

**Overview**:
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art deep learning model designed for natural language processing (NLP) tasks. It leverages the transformer architecture to understand the context of words in a sentence bidirectionally.

**Steps**:

- **Text Tokenization**: The BERT tokenizer is used to convert the text data into tokens that the BERT model can process. This involves breaking down the text into subwords and adding special tokens required by the BERT architecture.
- **Model Training**: The pre-trained BERT model is fine-tuned on our dataset of news articles. Fine-tuning involves updating the pre-trained model weights with our specific data to improve its performance on the fake news detection task.
- **Performance Evaluation**: The fine-tuned BERT model is evaluated using the same metrics as Logistic Regression. Additionally, we may use the ROC-AUC score to measure the model's ability to discriminate between fake and real news.

**Advantages**:

- Captures complex patterns and context in the text.
- Handles long-range dependencies and relationships between words.
- State-of-the-art performance on many NLP benchmarks.

**Limitations**:

- Computationally intensive and requires significant resources for training.
- Slower inference compared to traditional models like Logistic Regression.

### Comparison and Selection

Both models have their strengths and weaknesses. Logistic Regression offers simplicity and speed, making it suitable for scenarios where quick and interpretable results are needed. On the other hand, BERT provides superior performance by capturing the nuanced context of language, albeit at a higher computational cost.

**Evaluation Criteria**:

- **Accuracy**: The overall correctness of the model in classifying news articles.
- **Precision and Recall**: Balancing the trade-off between correctly identifying fake news and minimizing false positives.
- **F1 Score**: A comprehensive metric that considers both precision and recall.
- **Inference Time**: The time taken to classify a new article, important for real-time applications.

By evaluating both models on these criteria, we can determine the best approach for our fake news detection system, potentially using Logistic Regression for baseline performance and interpretability and BERT for high-accuracy and context-aware classification.

<h2> Website Interface </h2>

Our Fake News Detection website consists of 5 main pages. Below is an overview of each page with corresponding screenshots.

### 1. Home Page

The Home Page provides an introduction to our project, explaining its purpose and key features. It also includes a "Let's go for Prediction" button that redirects users to the Prediction page for analyzing news articles.

![Home Page](/static/Assets/Home_Page.png)

### 2. Prediction Page

In the Prediction Page, users can enter a news headline or article to check if it is real or fake. After clicking the "Get Result" button, users are redirected to the Analysis Results page.

![Prediction Page](/static/Assets/Prediction_Page.png)

### 3. Analysis Results

In the Analysis Results Page, users receive the prediction result of the news headline or article they entered. Additionally, based on the user's input, the page generates related articles using a news API.

![Analysis Results](/static/Assets/Result_Page.png)

### 4. Search News Page

The Search News Page serves as a utility tool where users can search for news articles related to specific topics. This feature helps users find and explore relevant news content easily.

![Search Page](/static/Assets/Search_Page.png)

### 5. About Us

The About Us Page contains some information about our project team, including team members' roles and contact details.

![About Us](/static/Assets/about_us.png)

<h2> Impact</h2>

By developing an effective fake news detection system, this project aims to contribute to the fight against misinformation. Such a system can be used by news organizations, social media platforms, and individual users to identify and mitigate the spread of fake news, ultimately promoting a more informed and truthful public discourse.
