import numpy as np
np.random.seed(0)
np.random.seed(1)
from lstm_model import *
from sentence_indices import *

##############
# Get data
##############
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

# Maximum length
maxLen = len(max(X_train, key=len).split())

##############
# Prepare Input
##############
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

##############
# Call & Compile Model
##############
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##############
# Train Model
##############
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

##############
# Evaluate Model: You should get a test accuracy between 80% and 95%.
##############
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

##############
# This code allows you to see the mislabelled examples##############
##############
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

##############
# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.
##############
x_test = np.array(["leave me alone"])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))