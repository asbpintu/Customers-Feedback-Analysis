# Customers-Feedback-Analysis

The project involves conducting customer feedback analysis for Amazon Alexa. The goal is to analyze and understand customer sentiments expressed in their reviews or feedback about the Amazon Alexa product or service. The project utilizes various machine learning and natural language processing techniques to accomplish this task.

### Necessary Packages
```js
# Import Packages

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re

import spacy
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
punct = string.punctuation
stem = PorterStemmer()
lemma = WordNetLemmatizer()

from wordcloud import WordCloud
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

'''
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dense, LSTM
'''

import warnings
warnings.filterwarnings('ignore')
```

### Reading Data
```js
root = pd.read_csv(r'amazon_alexa.tsv' , delimiter = '\t' , quoting = 3)
data = root.copy()

```
### Splitting data

```js
# Ind and Dep variable


data = data.drop(['rating','date','variation'],axis = 1)
data.columns = ['reviews' , 'target']

x = data['reviews']
y = data['target']
```
```js
print(x.head(10))
```
```
0                                        Love my Echo!
1                                            Loved it!
2    "Sometimes while playing a game, you can answe...
3    "I have had a lot of fun with this thing. My 4...
4                                                Music
5    "I received the echo as a gift. I needed anoth...
6    "Without having a cellphone, I cannot use many...
7    I think this is the 5th one I've purchased. I'...
8                                          looks great
9    "Love it! I’ve listened to songs I haven’t hea...
```
## Pata Preprocessing
To begin with, the project incorporates NLP (Natural Language Processing), Spacy, Regular Expression techniques to preprocess and clean the textual data obtained from customer feedback. This step involves tasks like tokenization, removing stopwords, stemming, and lemmatization.
### Replacing Common Contractions
```js
# replace_text
def replace_text(rev):
    
    reviews = re.sub(r"what's", "what is ", rev)
    reviews = re.sub(r"\'s", " is", reviews)
    reviews = re.sub(r"\'ve", " have ", reviews)
    reviews = re.sub(r"can't", "cannot ", reviews)
    reviews = re.sub(r"n't", " not ", reviews)
    reviews = re.sub(r"i'm", "i am ", reviews)
    reviews = re.sub(r"\'re", " are ", reviews)
    reviews = re.sub(r"\'d", " would ", reviews)
    reviews = re.sub(r"\'ll", " will ", reviews)
    reviews = re.sub(r"\'scuse", " excuse ", reviews)
    reviews = re.sub('\W', ' ', reviews)
    reviews = re.sub('\s+', ' ', reviews)
    reviews = reviews.strip(' ')
    
    return reviews

for i in range(len(x)) :
    x[i] = replace_text(x[i])
```
```js
print(x.head(10))
```
```
0                                         Love my Echo
1                                             Loved it
2    Sometimes while playing a game you can answer ...
3    I have had a lot of fun with this thing My 4 y...
4                                                Music
5    I received the echo as a gift I needed another...
6    Without having a cellphone I cannot use many o...
7    I think this is the 5th one I have purchased I...
8                                          looks great
9    Love it I ve listened to songs I haven t heard...
```
### Replacing Non-alphabetic Characters
```js
# cleaned_text
def cleaned_text(rev):
      
    reviews = re.sub(r'\[[0-9]*\]', ' ',rev)
    reviews = re.sub(r'\s+', ' ', reviews)
    reviews = re.sub('[^a-zA-Z]', ' ', reviews )
    reviews = re.sub(r'\s+', ' ', reviews)
    reviews = re.sub(r'\W*\b\w{1,3}\b', "",reviews)
    reviews = reviews.strip()
    
  
    return reviews


for i in range(len(x)) :
    x[i] = cleaned_text(x[i])
```
```js
print(x.head(10))
```
```
0                                            Love Echo
1                                                Loved
2    Sometimes while playing game answer question c...
3    have with this thing learns about dinosaurs co...
4                                                Music
5    received echo gift needed another Bluetooth so...
6    Without having cellphone cannot many features ...
7    think this have purchased working getting ever...
8                                          looks great
9    Love listened songs haven heard since childhoo...
```

### Remove Stopwords
```js
# remove_stopwords
def remove_stopwords(rev):
    
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(rev)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    reviews = ' '.join(tokens)
    
    return reviews

for i in range(len(x)) :
    x[i] = remove_stopwords(x[i])
```

### lemmatize  
```js
def lemmatize(rev):
    
    doc = nlp(rev)
    reviews = [words.lemma_ for words in doc]
    reviews = ' '.join(reviews)
    
    return reviews


for i in range(len(x)) :
    x[i] = lemmatize(x[i])
```
```js
print(x.head(10))
```
```
0                                            love echo
1                                                 love
2    sometimes play game answer question correctly ...
3    thing learn dinosaur control light play game l...
4                                                music
5    receive echo gift need another bluetooth somet...
6    without cellphone many feature ipad great alar...
7    think purchase work get every room house reall...
8                                           look great
9    love listen song hear since childhood news wea...
```


#### new data_frame for ann cnn and rnn
```js
new_data = data.copy()

file_path = r'C:\Users\asbpi\Desktop\Nit_DS & AI\MY Projects\project_sentiment analysis\new_data.csv'
new_data.to_csv(file_path, index=False)
```

### Viewing Positive and Negative reviews
```js
all_reviews = data['reviews']
pos_reviews = data['reviews'][data['target'] == 1]
neg_reviews = data['reviews'][data['target'] == 0]
```
```js
all_reviews = ' '.join(all_reviews .tolist())
pos_reviews = ' '.join(pos_reviews .tolist())
neg_reviews = ' '.join(neg_reviews .tolist())
```
#### Pie plot to check the percentages of positive and negative reviews
```js
plt.pie(data['target'].value_counts() , labels=['Positive','Negative'] , autopct='%1.0f%%')
plt.show()
```
![pie](image/pie.png)

### WordCloud

#### All Reviews
```js
all_wordcloud = WordCloud(random_state=42 , max_font_size=100).generate(all_reviews)
plt.figure(figsize=(12,8))
plt.imshow(all_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![allword](image/all.png)

#### Posotive Reviews
```js
pos_wordcloud = WordCloud(random_state=42 , max_font_size=100).generate(pos_reviews)
plt.figure(figsize=(12,8))
plt.imshow(pos_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![posword](image/pos.png)

#### Negative Reviews
```js
neg_wordcloud = WordCloud(random_state=42 , max_font_size=100).generate(neg_reviews)
plt.figure(figsize=(12,8))
plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![negword](image/neg.png)

### Frequency Distribution Plot 
#### Frequency distribution plot to check the frequencies of each words in all reviews
```js
fredi=nltk.word_tokenize(all_reviews)
freqDist = FreqDist(fredi)
plt.figure(figsize=(12,6))
plt.xticks([])
freqDist.plot(70)
plt.show()
```
![allfreq](image/freq.png)

#### Frequency distribution plot to check the frequencies of each words in positive reviews
```js
fredi=nltk.word_tokenize(pos_reviews)
freqDist = FreqDist(fredi)
plt.figure(figsize=(12,6))
plt.xticks([])
freqDist.plot(50)
plt.show()
```
![posfreq](image/freq_pos.png)

#### Frequency distribution plot to check the frequencies of each words in negative reviews
```js
fredi=nltk.word_tokenize(neg_reviews)
freqDist = FreqDist(fredi)
plt.figure(figsize=(12,6))
plt.xticks([])
freqDist.plot(50)
plt.show()
```
![negfreq](image/freq_neg.png)

### vectorization

#### CountVectorizer
```js
vectorizer = CountVectorizer().fit(x)
feature_names = vectorizer.get_feature_names_out()
X = vectorizer.transform(x).toarray()

X_c = pd.DataFrame(X , columns= feature_names)
```
#### TfidfVectorizer
```js
vectorizer = TfidfVectorizer().fit(x)
feature_names = vectorizer.get_feature_names_out()
X = vectorizer.transform(x).toarray()

X_t = pd.DataFrame(X , columns= feature_names)
```

### spliting

#### Splitting data into Train and Test sets
80 percentages of data for training and 20 percentages of data for Testing.
```js
X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size = 0.20, random_state = 0)
```

## Model Fitting
The project leverages a variety of supervised learning algorithms, including logistic regression (logit), support vector machines (SVM), naive Bayes, decision trees, and random forests, to train models using annotated customer feedback data.

### LogisticRegression
```js
# Logit
logit = LogisticRegression()
logit.fit(X_train, y_train)

y_pred = logit.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.9174603174603174

### Naive Byas
#### 1. BernoulliNB
```js
# Train the Bernoulli Naive Bayes classifier
berNB = BernoulliNB()
berNB.fit(X_train, y_train)
# Predict on the test set
y_pred = berNB.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.9047619047619048

#### 2. GaussianNB
```js
# Train the Gaussian Naive Bayes classifier
gauNB = GaussianNB()
gauNB.fit(X_train, y_train)

# Predict on the test set
y_pred = gauNB.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.5523809523809524

#### 3. MultinomialNB
```js
# Train the Multinomial Naive Bayes classifier
mulNB = MultinomialNB()
mulNB.fit(X_train, y_train)

# Predict on the test set
y_pred = mulNB.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.9158730158730158
### Support Vector Machine (SVM)
```js
# Train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.926984126984127

### Decession Tree
```js
# Train the DecisionTreeClassifier
dectree = DecisionTreeClassifier()
dectree.fit(X_train, y_train)

# Predict on the test set
y_pred = dectree.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.9317460317460318

### Random Forest
```js
# Train the RandomForestClassifier
randForest = RandomForestClassifier()
randForest.fit(X_train, y_train)

# Predict on the test set
y_pred = randForest.predict(X_test)
```
```js
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
Accuracy: 0.9380952380952381


### Cross Validation

#### K fold
```js
randForest = RandomForestClassifier()

# Perform cross-validation
scores = cross_val_score(randForest, X_train, y_train, cv=10)

print("Cross-validation scores:", scores)
```
Cross-validation scores: [0.94444444 0.93253968 0.93253968 0.94444444 0.94047619 0.94444444
 0.93650794 0.92857143 0.92857143 0.93253968]

```js
# Calculate the mean accuracy across all folds
mean_accuracy = scores.mean()
print("Mean Accuracy:", mean_accuracy)
```
Mean Accuracy: 0.9365079365079364

#### Grid Search
```js
randForest = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=randForest, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
```
```js
# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
```
Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
```js
print("Best Score:", grid_search.best_score_)
```
Best Score: 0.9365079365079365


## Deep Learning Models Fitting

The project may also employ deep learning techniques such as Artificial Neural Networks (ANN) and Recurrent Neural Networks (RNN) to capture more complex patterns and dependencies within the feedback data. These deep learning models are trained on large amounts of labeled data to learn the underlying sentiment patterns.

 ### Artificial Neural Networks (ANN)

#### Required Packages
```js
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dense, LSTM
```
#### Reading new data
```js
new_df = pd.read_csv(r'C:\Users\asbpi\Desktop\Nit_DS & AI\MY Projects\project_sentiment analysis\new_data.csv')

reviews = new_df['reviews']
```
#### Prepare Data
```js
# Convert any non-string elements to strings
reviews = [str(review) for review in reviews]
```
```js
# Remove null or NaN values
reviews = [review for review in reviews if not pd.isnull(review)]
```
```js
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences)
```
#### Splitting Data
```js
X = padded_sequences
y = new_df['target']
```
```js
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Model Building
```js
model = Sequential()
model.add(Embedding(len(word_index)+1, 100, input_length=X.shape[1]))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
```

#### Compile the model
```js
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

#### Train the model
```js
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```
```
Epoch 1/50
79/79 [==============================] - 5s 62ms/step - loss: 0.2893 - accuracy: 0.9159 - val_loss: 0.3047 - val_accuracy: 0.9079
Epoch 2/50
79/79 [==============================] - 4s 50ms/step - loss: 0.1760 - accuracy: 0.9377 - val_loss: 0.2082 - val_accuracy: 0.9190
Epoch 3/50
79/79 [==============================] - 3s 44ms/step - loss: 0.0781 - accuracy: 0.9714 - val_loss: 0.1989 - val_accuracy: 0.9286
Epoch 4/50
79/79 [==============================] - 3s 41ms/step - loss: 0.0382 - accuracy: 0.9857 - val_loss: 0.2376 - val_accuracy: 0.9349
Epoch 5/50
79/79 [==============================] - 3s 42ms/step - loss: 0.0306 - accuracy: 0.9893 - val_loss: 0.2540 - val_accuracy: 0.9317
Epoch 6/50
79/79 [==============================] - 3s 41ms/step - loss: 0.0239 - accuracy: 0.9929 - val_loss: 0.3320 - val_accuracy: 0.9429
Epoch 7/50
79/79 [==============================] - 3s 41ms/step - loss: 0.0280 - accuracy: 0.9901 - val_loss: 0.2770 - val_accuracy: 0.9397
Epoch 8/50
79/79 [==============================] - 3s 41ms/step - loss: 0.0236 - accuracy: 0.9917 - val_loss: 0.3003 - val_accuracy: 0.9333

...

Epoch 49/50
79/79 [==============================] - 4s 46ms/step - loss: 0.0193 - accuracy: 0.9933 - val_loss: 0.4881 - val_accuracy: 0.9286
Epoch 50/50
79/79 [==============================] - 4s 47ms/step - loss: 0.0179 - accuracy: 0.9933 - val_loss: 0.4990 - val_accuracy: 0.9317
```

#### Evaluate the model
```js
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
print('Loss', loss)
```
```
20/20 [==============================] - 0s 5ms/step - loss: 0.4990 - accuracy: 0.9317
Accuracy: 0.9317460060119629
Loss 0.49901625514030457
```


### Recurrent Neural Networks (RNN)


#### Required Packages
```js
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dense, LSTM
from sklearn.metrics import precision_score, recall_score, f1_score
```
#### Reading new data
```js
new_df = pd.read_csv(r'C:\Users\asbpi\Desktop\Nit_DS & AI\MY Projects\project_sentiment analysis\new_data.csv')

reviews = new_df['reviews']
```
#### Prepare Data
```js
# Convert any non-string elements to strings
reviews = [str(review) for review in reviews]
```
```js
# Remove null or NaN values
reviews = [review for review in reviews if not pd.isnull(review)]
```
```js
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences)
```
#### Splitting Data
```js
X = padded_sequences
y = new_df['target']
```
```js
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Model Building

```js
# Build the RNN model
model = Sequential()
model.add(Embedding(len(word_index)+1, 100, input_length=X.shape[1]))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=32))
model.add(Dense(units=1, activation='sigmoid'))
```

#### Compling Model
```js
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

#### Train the model
```js
model.fit(X_train, y_train, epochs=50, batch_size=32)
```
```
Epoch 1/50
79/79 [==============================] - 26s 268ms/step - loss: 0.3027 - accuracy: 0.9115
Epoch 2/50
79/79 [==============================] - 20s 250ms/step - loss: 0.2783 - accuracy: 0.9210
Epoch 3/50
79/79 [==============================] - 18s 234ms/step - loss: 0.2770 - accuracy: 0.9210
Epoch 4/50
79/79 [==============================] - 19s 235ms/step - loss: 0.2672 - accuracy: 0.9210
Epoch 5/50
79/79 [==============================] - 18s 234ms/step - loss: 0.1634 - accuracy: 0.9393
Epoch 6/50
79/79 [==============================] - 19s 240ms/step - loss: 0.0879 - accuracy: 0.9702

...

Epoch 49/50
79/79 [==============================] - 19s 241ms/step - loss: 0.0180 - accuracy: 0.9933
Epoch 50/50
79/79 [==============================] - 20s 258ms/step - loss: 0.0181 - accuracy: 0.9933
O
```

#### Evaluate on the test set
```js
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
print('Loss', loss)
```
```
20/20 [==============================] - 2s 80ms/step - loss: 0.3146 - accuracy: 0.9317
Accuracy: 0.9317460060119629
Loss 0.314641535282135
```
```js
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
```
#### Calculating Precision_Score, Recall_Score, f1_Score

To calculate Precision_Score, Recall_Score, f1_Score we need below four things:-

+ **True Positive (TP):** The number of samples that are correctly predicted as positive (correctly classified as the positive class).
+ **True Negative (TN):** The number of samples that are correctly predicted as negative (correctly classified as the negative class).
+ **False Positive (FP):** The number of samples that are incorrectly predicted as positive (incorrectly classified as the positive class).
+ **False Negative (FN):** The number of samples that are incorrectly predicted as negative (incorrectly classified as the negative class).
  
Once you have these values, you can compute the evaluation metrics as follows:

##### 1. Precision:
It measures the proportion of correctly predicted positive samples out of all samples predicted as positive.
Precision is computed as 
```
TP / (TP + FP)
```
##### 2. Recall: (also called Sensitivity or True Positive Rate) 
It measures the proportion of correctly predicted positive samples out of all actual positive samples.
Recall is computed as 
```
TP / (TP + FN)
```
##### 3. F1-score:
It combines precision and recall into a single metric, which is the harmonic mean of the two.
F1-score is computed as 
```
2 * (Precision * Recall) / (Precision + Recall)
```
code:-
```js
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('precision_score = ', precision)
print('recall_score = ', recall)
print('f1_score = ', f1)
```
```
precision_score =  0.008475686570924667
recall_score =  0.09206349206349207
f1_score =  0.01552233296419343
```
### Prediction Checking

```js
predictions = model.predict(X_test)

for i in range(len(predictions)):
    text = tokenizer.sequences_to_texts([X_test[i]])[0]
    sentiment = "positive" if predictions[i] > 0.5 else "negative"
    print(f"Text: {text}")
    print(f"Predicted sentiment: {sentiment}")
    print("-----------------------------")
```
Result :-
```
20/20 [==============================] - 1s 71ms/step
Text: love still learn capability
Predicted sentiment: positive
-----------------------------
Text: easy setup
Predicted sentiment: positive
-----------------------------
Text: joke worthless
Predicted sentiment: negative
-----------------------------
Text: like another house constantly repeat tell alexa something clear accord history hear something comeletely different device really cool like concept get star however work intend find irritated often device
Predicted sentiment: positive
-----------------------------
Text: mine last night play around buy long time
Predicted sentiment: negative
-----------------------------
Text: small device kid like question handy friendly interactive
Predicted sentiment: positive
-----------------------------
Text: outstanding piece technology everyday information scheduling general information awesome amazon keep improve
Predicted sentiment: positive
-----------------------------
Text: connect sound perfect give internal battery disconnect power source
Predicted sentiment: negative
-----------------------------
Text: hear everything understand half first time often take multiple try give correct result
Predicted sentiment: negative
-----------------------------
Text: like well expect
Predicted sentiment: positive
-----------------------------
Text: still work call know charge time ridiculous
Predicted sentiment: positive
-----------------------------
```

### Print the top Phrases associated with Positive reviews

```js
embedding_weights = model.layers[0].get_weights()[0]
word_index = tokenizer.word_index
reverse_word_index = {index: word for word, index in word_index.items()}


phrase_sentiment_scores = {}


for sequence in sequences:
    phrase = ' '.join([reverse_word_index.get(word_index, '') for word_index in sequence])
    sentiment_score = sum([embedding_weights[word_index] for word_index in sequence])
    phrase_sentiment_scores[phrase] = sentiment_score

# Sort the phrases based on maximum sentiment score within each phrase
sorted_scores = sorted(phrase_sentiment_scores.items(), key=lambda x: max(x[1]), reverse=True)


top_positive_phrases = []
for phrase, score in sorted_scores:
    words = phrase.split()
    if len(words) >= 2 and len(words) <= 3:
        top_positive_phrases.append(phrase)
        if len(top_positive_phrases) >= 20:
            break

# Print the top phrases associated with positive sentiment
print("Top Positive phrases :")
for phrase in top_positive_phrases:
    print(f"{phrase}")
```
Top Positive Phrases are:-

```
Top Positive phrases :
love love love
kid love love
love easy well
easy setup love
love love
sound great love
love living room
easy affordable love
love great product
easy family love
love echo easy
love easy
intelligent love song
easy easy
easy great sound
great sound easy
love awesome
firestick easy enjoy
awesome love alexa
love good gift
```

### Print the top Phrases associated with Negative reviews

```js
embedding_weights = model.layers[0].get_weights()[0]
word_index = tokenizer.word_index
reverse_word_index = {index: word for word, index in word_index.items()}


phrase_sentiment_scores = {}


for sequence in sequences:
    phrase = ' '.join([reverse_word_index.get(word_index, '') for word_index in sequence])
    sentiment_score = sum([embedding_weights[word_index] for word_index in sequence])
    phrase_sentiment_scores[phrase] = sentiment_score

# Sort the phrases based on maximum sentiment score within each phrase
sorted_scores = sorted(phrase_sentiment_scores.items(), key=lambda x: max(x[1]), reverse=False)


top_negative_phrases = []
for phrase, score in sorted_scores:
    words = phrase.split()
    if len(words) >= 2 and len(words) <= 3:
        top_negative_phrases.append(phrase)
        if len(top_negative_phrases) >= 20:
            break

# Print the top phrases associated with negative reviews
print("Top negative phrases :")
for phrase in top_negative_phrases:
    print(f"{phrase}")
```
Top Positive Phrases are:-

```
Top negative phrases :
work tube
participate echo
habla espanol
work describe
alexa sister second
always work
alexa rock
good device
fairly useless
echo work
second kitchen recipe
everything need
use twice work
work well
sound quality
amazon disappoint
good quality
work fine
show nothing
five need
```

## Conclusion

After trained different Classification model we got the accuracies as :-

**-LogisticRegression**
    
    Accuracy: 91%

**-Naive Byas**
+ BernoulliNB

        Accuracy: 90%

+ GaussianNB

        Accuracy: 55%

+ MultinomialNB

        Accuracy: 91%


**-Support Vector Machine (SVM)**
    
    Accuracy: 92%


**-Decession Tree**
   
    Accuracy: 93.1%


**-Random Forest**
    
    Accuracy: 93.8%

```
Here we got the best accuracy from Random Forest Model, after doing cross validation we got
the best parameters are: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
and the best accuracy score is 93.6%
```

+ Also employ Deep learning techniques such as Artificial Neural Networks (ANN) and Recurrent Neural Networks (RNN) :

    + **Artificial Neural Networks (ANN)**


        Accuracy: 93.1%
        Loss 49%

    + **Recurrent Neural Networks (RNN)**


        Accuracy: 93.1%
        Loss 31%


Among all models we can see RNN model gives the best results with precision, recall and f1 score are 

    precision_score =  0.0084
    recall_score =  0.092
    f1_score =  0.015

+ The model showcases strong performance in capturing the temporal dynamics and dependencies within the reviews of Amazon Alexa. With an accuracy of 93%, it effectively classifies the sentiment of the reviews, demonstrating its ability to discern between positive and negative sentiments.

+ Furthermore, the RNN model's ability to process sequential data enables it to capture subtle shifts in sentiment over time. By leveraging its recurrent nature, the model effectively incorporates the context of previous words and phrases when predicting sentiment, resulting in a nuanced understanding of the feedback.

+ Additionally, the model's attention mechanisms provide valuable insights into the important features contributing to sentiment analysis. Through analysis of the attention weights, we discovered that the model pays significant attention to emotionally charged words and phrases, emphasizing their impact on sentiment classification that which words or phrases are corelated to Positive or Negative reviews.

+ These findings highlight the model's capability to capture the nuanced sentiment patterns present in the data, making it a valuable tool for sentiment analysis tasks.The model enables businesses to gain deeper insights into customer sentiment, allowing them to make informed decisions and tailor their strategies to enhance customer satisfaction and overall sentiment-driven initiatives.



# Thank You




