import pandas as pd
from baseline_model_functions import *

emoji_dictionary = {"0": "\u2764\uFE0F",  # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
##############
# Get data
##############
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# Maximum length
maxLen = len(max(X_train, key=len).split())

# Print sentences and related emoji
print("Print sentences and related emoji")
for idx in range(15):
    print(X_train[idx], label_to_emoji(Y_train[idx]))

# Get one hot encoded Y
Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)

# Example
idx = 2
print(f"Sentence '{X_train[idx]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")

##############
# Read GloveVectors
##############
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# Example
word = "cucumber"
idx = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])

##############
# Train Model!
##############
np.random.seed(1)
pred, W, b = model(X_train, Y_train, word_to_vec_map)
print("Training is completed!")

##############
# Predict!
##############
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

# Example Predictions
print("Example Predictions")
X_my_sentences = np.array(["i cherish you", "i love you", "funny lol", "lets play with a ball", "food is ready",
                           "not feeling happy", "not okay", "lack of happiness", "I am not sorry.", "Yo don't look fine.",
                          "worst day ever", "best day ever", "my heart belongs to you", "not funny"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)


##############
# Classification Report
##############
print("Classification Report")
print(Y_test.shape)
print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)


