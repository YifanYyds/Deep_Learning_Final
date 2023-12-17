# Comparing LSTM and Simple RNN for Sentiment Analysis
## Introduction
In this project, we explore the performance differences between Long Short-Term Memory (LSTM) networks and Simple Recurrent Neural Networks (RNNs) in the context of sentiment analysis. Utilizing TensorFlow Keras, we implement and compare these models to understand their strengths and limitations in handling sequential data. Our focus is on evaluating their accuracy and training efficiency, providing insights into their applicability for various tasks in deep learning.
## Dataset
This project utilizes a publicly available dataset for multiclass sentiment analysis, which can be accessed [here](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset).
 The dataset is pre-divided into training, validation, and test sets, simplifying the data preparation process. It consists of textual data classified into three sentiment categories: negative, neutral, and positive, represented numerically as 0, 1, and 2, respectively. This rich dataset provides a realistic scenario for evaluating the comparative effectiveness of LSTM and Simple RNN models in classifying sentiments with varying degrees of complexity.
## Data preprocessing
For both the LSTM and Simple RNN models, we employed a uniform data preprocessing pipeline to ensure consistency in our comparison. The preprocessing steps included:

Tokenization: We used Keras's Tokenizer to convert text data into sequences of integers, where each integer represents a specific word in a dictionary created from the entire training text.

Sequence Padding: Given the variable length of text data, we padded all sequences to a uniform length, determined by the longest sequence in the dataset, using Keras's pad_sequences method.

One-Hot Encoding of Labels: The labels, representing three sentiment classes (negative, neutral, and positive), were one-hot encoded for compatibility with our model's output layer, facilitating a multi-class classification task.

Data Splitting: The dataset was already divided into training, validation, and testing sets, allowing for a structured training and evaluation process.

Extraction of Subsets: For detailed analysis, we extracted subsets of the test data corresponding to each of the three sentiment classes using their one-hot encoded labels.

This preprocessing approach ensured that the input data to both models was standardized, allowing for a direct comparison of their performance.
## Model Description
## Hyperparameter Tuning
## Callbacks
## Results
## Source code
## Trained Model
The trained model can be accessed and downloaded from the following Google Drive folder: [Trained Model Folder](https://drive.google.com/drive/folders/1T2XThXZVQo-NYV6WD0_FdAB8pZ6qOYrb?usp=sharing)

Please refer to the folder for the model files. Instructions for loading and using the model are provided below.
## References

