# Machine_Learning_News_Classification
## Overview
It is a fantastic project that I learned in McGill continuing study! 
And This is a supervised learning task for Natural Language Processing, consisting of the headline and title of NEWS collected from several sources. 
The NEWS is classified into 4 categories:  
1. International  
2. Games  
3. Commerce  
4. Science&Technology   

More detail would be available in kaggle: https://www.kaggle.com/c/hw2-ycbs-273-intro-to-prac-ml/overview  

## Acknowledge
Before starting, I would like to thank my lecturer Arbaaz Khan for his patience and guidance. It was because of you that I was exposed to the wide world of Machine Learning.

## Introduction
In this place, the reason why I choose Bidirectional LSTM modeling rather than RNN, LSTM and so on would be shown. And the impact of different hyperparameters on model 
would be discussed. The methods to adjust parameter I found when I trained the model could possibly be benefial to you.

## 1. Model Selecting
### 1.1 Why Learning NLP Model?
In this time, The objective is to categorize the news, and simply using Neural Network like 'Input Layer -- Hidden Layers -- Output Layer' could
not have a good prediction, since every single words in a sequence is probably influenced by each other and we need to predict
based on the context. For example, there is a sentence, let's say 'Tom ate an apple.', and we expect the model could predict the characteristic or
property of a certain word. Spontaneously, the word 'apple' should have a higher probability of being noun since 'ate' is a verb.  

### 1.2 N-Grams Model  
This model has been out of date already for a long time. The main reason is that it is not sensible to only consider N words in prediction. 
For example, we want to do gap filling, and let's say 'Tom got an apple yesterday, and he do not like that ___ somehow.' If we choose 2-Gram Modeling,
the neural networks would only take 'that' into account, which is not reasonable anyway. Even if you set a much higher N, on the one hand, the sentence 
expected to be predicted would be different so the N is almost impossible to decided in advanced, on the other hand, the higher the N is, the more complex 
the neural network is, which is not practical.

### 1.3 Bag of Words Model
At the very beginning, I thought BOW model would be very suitable for this project, since the News is classified into 4 categories, and it is possible that 
each category has some subject specific. But after reading some samples in test data, I found that some sentences are not very characteristical so 
I changed my mind and decide to move to the sequence modeling. By the way, the loss based on BOW Model is good but not the best.

### 1.4 RNN Model
RNN is actually good in many cases, but when the sequences are very long, RNN modeling would have the problem of Exploding Gradients and Vanishing Gradients. 
The mathematical process of calculating gradients would not be shown here. You can check it in the passage below.  
General Concept(In Chinese): https://zybuluo.com/hanbingtao/note/541458    
Exploding Gradients: https://machinelearningmastery.com/exploding-gradients-in-neural-networks/  
Vanishing Gradients: https://en.wikipedia.org/wiki/Vanishing_gradient_problem  

### 1.5 LSTM & GRU Model
LSTM successfully solve the problem of common RNN, which makes it one of the most popular NLP neural network in the recent time. The mathematical process of 
LSTM is really complex, and it takes a lot to explain. If you are intereted in it, you can get a view on it in webesite below.  
LSTM Algorithm(In Chinese): https://zybuluo.com/hanbingtao/note/581764  

### 1.6 Bidirectional LSTM (Bilstm)
Bidirectional LSTM is actually a more advanced model compared to the LSTM model. One problem with LSTM modeling is that it cannot encode information 
from back to front, whereas BilSTM can. 

### 1.7 Conclusion
According to the characters of different models we discuss above and the experiment performance I got from each model. The Bidirectional LSTM was chosen finally.

## 2.The impact of different Hyperparameters
### 2.1 Learning Rate
The learning rate is extremely important in model training. To find out the optimal solution, the learning rate had better be low since a large learning rate 
could possibly pass the best performace. When I trained the model, I found a good way to set learning rate. Firstly, you can set a high learning rate like **5e-4** 
at the very beginning, and set a high **epochs** to see when would the loss reach the valley bottom. Then reset the model, and train again. Set a relatively high 
learning rate like 5e-4 and when you notice the loss almost reaches the lowest point you saw in the last step, you can stop training and compile a much lower learning rate 
like 5e-6 for next train(NO need to reset the model! Just Compile again.). By this method, you probably get the minimum eventually.   
The method above is just my experience, and I also read some useful passage. I will put it below.  
*How to Configure the Learning Rate: https:* //machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/   
*Machine Learning-Leaning rate:(In chinese)*  https://blog.csdn.net/rlnLo2pNEfx9c/article/details/80237586   

### 2.2 LayerNormalization & BatchNormalization
According to my training records, the LayerNormalization is really important for a model to converge steadily. And some passages I read argued that for NLP, 
the LayerNormalization could be better because BatchNormalization is more sensitive to the batch size since in NLP the length of sequences could be totally different. 
So in order to reduce the degree of over fitting, a LayerNormalization after each hidden layer is really significant .    
More Detial on: https://blog.csdn.net/liuxiao214/article/details/81037416  

### 2.3 Embedding Layer\*
Embedding Layer is the most important one! Before adding the embedding layer, It was really hard for the model to reach a better performance. But after using **pre-trained** 
Embedding layer (glove.6B.100d.txt), the model straightly be to the moon! So we could realize that a dictionary for student who are learning English is really helpful, 
just like the embedding layer for the model. (Actually you can regard the Embedding Layer like a look-up table)  
More detial on: https://blog.csdn.net/qq_36523492/article/details/112260687  

### 2.4 max_tokens & max_length in Vectorization
Though max_tokens & max_length have a slightly significant impact on the model training, it is helpful. And it is easy to understand, the larger the dictionary is, the more 
words the model could detect.  
