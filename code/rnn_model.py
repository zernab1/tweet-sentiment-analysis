import numpy as np
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Import data
nltk.download('twitter_samples')
nltk.download('stopwords')
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

tk = TweetTokenizer()
stop_words = set(stopwords.words('english'))

def preprocess_tweet(tweet):
    tokens = tk.tokenize(tweet)
    cleaned_tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return " ".join(cleaned_tokens)

positive_tweets = [preprocess_tweet(tweet) for tweet in positive_tweets]
negative_tweets = [preprocess_tweet(tweet) for tweet in negative_tweets]

tweets = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

# Tokenize and Pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweets)

X = tokenizer.texts_to_sequences(tweets)
X = pad_sequences(X, maxlen=100)  # Adjust maxlen based on your text length

# The RNN Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=100))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train/Test split
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X, labels, test_size=0.2, random_state=42)

model.fit(X_train_rnn, np.array(y_train_rnn), epochs=5, batch_size=32, validation_split=0.2)

# Model Evaluation 
loss, accuracy = model.evaluate(X_test_rnn, np.array(y_test_rnn))
print(f"Accuracy: {accuracy:.2f}")

# Predictions
predictions = model.predict(X_test_rnn)

# Printing random tweets and their predicted/actual labels
for i in range(10):
    tweet_index = np.random.randint(len(X_test_rnn))
    tweet = tokenizer.sequences_to_texts([X_test_rnn[tweet_index]])[0]
    actual_label = y_test_rnn[tweet_index]
    predicted_label = round(predictions[tweet_index][0])

    print(f"\nTweet: {tweet}")
    print(f"Actual Label: {'Positive' if actual_label == 1 else 'Negative'}")
    print(f"Predicted Label: {'Positive' if predicted_label == 1 else 'Negative'}")
    print("-" * 50)