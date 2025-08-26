import random
import json
import nltk
from nltk.stem import PorterStemmer

# Download NLTK resources if not already present
nltk.download("punkt", quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

# Load intents
with open("../data/intents.json") as f:
    intents = json.load(f)

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem_words(words):
    return [stemmer.stem(word) for word in words]

def get_response(user_input):
    tokens = stem_words(tokenize(user_input))
    
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_tokens = stem_words(tokenize(pattern))
            if set(pattern_tokens).issubset(set(tokens)):
                return random.choice(intent["responses"])
    
    return "Sorry, I didn’t quite understand that."

if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye! 👋")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
