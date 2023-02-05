# Emojifyer

I've taken the lecture [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/3) from 
Coursera given by DeepLearning.AI. This project is prepared by the homework 
on week2. 

This model will be turn this sentence: 

"Congratulations on the promotion! Let's get coffee and talk. Love you!"

to: 

"Congratulations on the promotion!👍 Let's get coffee and talk.☕️ Love you!❤️"

We will implement two models:

    1- Baseline Model
    2- LSTM Model
    

## Baseline Model: Emojify-V1

- Get the average of embedding of each word in a sentence. 
- Apply softmax function and find the label 

Because adore has a similar embedding as love, the algorithm has generalized correctly even to a word it has never seen before.
Words such as heart, dear, beloved or adore have embedding vectors similar to love.

This algorithm ignores word ordering, so is not good at understanding phrases like "not happy."
Note that the model doesn't get the following sentence correct:

"not feeling happy"

## LSTM Model: Emojify-V2

This model will use the pretrained embdedding layer and insert it to LSTM model. It is many to one architecture. This 
means that:

- take the given sentence: "I love you."
- get the indices: 

The Emojify-V1 model did not "not feeling happy" correctly, but implementation of Emojify-V2 got it right!

If it didn't, be aware that Keras' outputs are slightly random each time, so this is probably why.

The current model still isn't very robust at understanding negation (such as "not happy")
This is because the training set is small and doesn't have a lot of examples of negation.

If the training set were larger, the LSTM model would be much better than the Emojify-V1 model at understanding more 
complex sentences.


## NOTES
What you should remember:

- If you have an NLP task where the training set is small, using word embeddings can help your algorithm significantly.
- Word embeddings allow your model to work on words in the test set that may not even appear in the training set.
- Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details:
    - To use mini-batches, the sequences need to be padded so that all the examples in a mini-batch have the same length.
    - An Embedding() layer can be initialized with pretrained values.
        - These values can be either fixed or trained further on your dataset.
        - If however your labeled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings.
- LSTM() has a flag called return_sequences to decide if you would like to return every hidden states or only the last one.
- You can use Dropout() right after LSTM() to regularize your network.


## Acknowledgments
Thanks to Alison Darcy and the Woebot team for their advice on the creation of this assignment.

- Woebot is a chatbot friend that is ready to speak with you 24/7.
- Part of Woebot's technology uses word embeddings to understand the emotions of what you say.
- You can chat with Woebot by going to http://woebot.io



