from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training messages
messages = [
    "Congratulations you won a prize",
    "You have won 50000 rupees",
    "Click this link to claim your reward",
    "Win cash now click here",
    "You are selected for a free gift",
    "Get free recharge now",
    "Congratulations claim your lottery prize",
    "Exclusive offer buy now",
    "Are you coming to college today",
    "Let's meet at 5 pm",
    "Can you send me the notes",
    "What time is the class",
    "I will call you later",
    "Where are you",
    "Please submit the assignment",
    "See you tomorrow"
]

# 1 = Spam, 0 = Not Spam
labels = [
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0
]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

# Train the AI model
model = LogisticRegression()
model.fit(X, labels)


def detect_spam(message):
    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)[0]
    probability = model.predict_proba(message_vector)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return result, round(confidence, 2)