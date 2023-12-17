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
### LSTM Model
#### Intro to LSTM
Long Short-Term Memory (LSTM) networks are a type of  Recurrent Neural Network (RNN) specialized in remembering information over extended sequences, which makes them suitable for tasks like sentiment analysis, language modeling, and sequence prediction.
#### Achitecture details
Embedding Layer: The LSTM model begins with an Embedding layer, which plays a crucial role in text processing tasks. This layer maps each word in the input sequence to a high-dimensional vector space, capturing semantic relationships between words. The size of the embedding space is a key hyperparameter, directly influencing the model's ability to encapsulate word relationships.

LSTM Layer: At the heart of the model lies the LSTM layer, which processes the sequence of word embeddings. Each LSTM unit comprises a complex arrangement of gates - the input, forget, and output gates. These gates collaboratively decide which information should be remembered and which should be discarded, enabling the network to maintain a memory over input sequences. The number of units in the LSTM layer, a critical hyperparameter, determines the capacity of the model to learn and store information. More units can provide a richer understanding of the sequence context but at the cost of increased computational complexity.

Output Layer: The model concludes with a Dense layer featuring a softmax activation function. This layer translates the LSTM's output into probabilities across the target classes. In the context of sentiment analysis, these classes typically correspond to different sentiment labels like negative, neutral, and positive.
### Simple RNN Model
#### Intro to Simple RNN
The Simple Recurrent Neural Network (RNN) is a fundamental type of neural network designed for processing sequential data. It's characterized by its straightforward architecture where the output from the previous step is fed back into the network at each step, allowing it to maintain a form of 'memory' of past inputs. This feature makes Simple RNNs suitable for tasks that involve sequences, such as language modeling and text classification. However, they are typically less complex and have fewer parameters than more advanced networks like LSTMs.
#### Architecture details
Embedding Layer: Similar to the LSTM model, the Simple RNN model also starts with an Embedding layer. This layer converts input text data into dense vectors of fixed size. Each word is represented by a vector in a predefined embedding space, capturing essential semantic relationships.

Simple RNN Layer: The core of this model is the Simple RNN layer, which processes the sequence of word embeddings. Unlike LSTMs, Simple RNNs do not have complex gating mechanisms like input, forget, and output gates. Instead, they rely on a simpler structure where the hidden state from the previous timestep is combined with the current input to produce the next output. This simplicity can lead to challenges such as the vanishing gradient problem, particularly in long sequences. The number of units in the Simple RNN layer is a key hyperparameter, influencing the model's ability to process and remember information across the sequence.

Output Layer: The model concludes with a Dense layer with a softmax activation, similar to the LSTM model. This layer outputs the probabilities for each class in the classification task, such as the different sentiments in sentiment analysis.

## Hyperparameter Tuning
Methodology
We utilized the Hyperband optimization algorithm, a variant of random search, with a novel early-stopping strategy. Hyperband speeds up the hyperparameter search process by adaptively allocating resources and halting less promising trials, thereby focusing on more promising configurations.

Tuned Hyperparameters
Embedding Output Dimension: This parameter determines the size of the embedding vectors generated by the Embedding layer. A larger dimension means the model can potentially capture more information about each word but also increases computational complexity.

Number of RNN Units: For both LSTM and Simple RNN models, we tuned the number of units in the LSTM/RNN layer. This parameter affects the capacity of the model to learn patterns in the data. More units can allow the model to capture more complex relationships but also increase the risk of overfitting and computational cost.

Tuning Process
We defined a range for each hyperparameter, allowing the tuner to explore different combinations within these ranges.
The tuner evaluated various configurations by training different versions of the model on the training data and assessing their performance on the validation data.
The objective was set to maximize validation accuracy.

## Callbacks
Early Stopping: To prevent overfitting and to expedite the tuning process, we employed an EarlyStopping callback. This callback monitors the model's validation loss and halts training if the model ceases to improve.

TimeHistory
Purpose: Measures the average training time per batch, providing insights into the training efficiency of the model.
Implementation:
At the beginning of each batch, the current time is recorded.
At the end of each batch, the time elapsed since the batch started is computed and stored.
At the end of each epoch, the average time per batch is calculated and appended to a global list, global_epoch_times.
This data can be used to compare the training efficiency of different models and configurations.

## Results
Our comparative analysis of LSTM and Simple RNN models revealed distinct differences in performance and efficiency. The LSTM model achieved a higher accuracy of XX% on the test set, compared to XX% for the Simple RNN, indicating its superior capability in handling complex sequential data. The TimeHistory callback analysis showed that the LSTM model had a slightly higher average training time per batch, which aligns with its more complex architecture.

| Model     | Test Accuracy | Avg. Training Time per Batch(sec) |
|-----------|----------|------------------------------|
| LSTM      | 65.89%      | 0.06292901959745524s                           |
| Simple RNN| 49.08%      | 0.3371764089771449s                           |
For detailed training logs, epoch-wise performance metrics, and additional plots, please refer to our comprehensive [Jupyter notebooks](https://github.com/YifanYyds/Deep_Learning_Final/blob/main/SimpleRNN_final.ipynb) and [Jupyter notebooks](https://github.com/YifanYyds/Deep_Learning_Final/blob/main/LSTM_final.ipynb)
## Source code
## Trained Model
The trained model can be accessed and downloaded from the following Google Drive folder: [Trained Model Folder](https://drive.google.com/drive/folders/1T2XThXZVQo-NYV6WD0_FdAB8pZ6qOYrb?usp=sharing)

Please refer to the folder for the model files. Instructions for loading and using the model are provided below.
## References
https://www.researchgate.net/publication/350543179_Sentiment_Analysis_Using_Deep_Learning
