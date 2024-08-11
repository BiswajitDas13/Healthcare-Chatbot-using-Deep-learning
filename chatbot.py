import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    print(f"Bag of words: {bow}")
    res = model.predict(np.array([bow]))[0]
    print(f"Model prediction: {res}")

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry, I do not understand your symptoms. Could you please rephrase?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I do not understand your symptoms. Could you please rephrase?"

def calling_the_bot(txt):
    predict = predict_class(txt)
    print(f"Predicted classes: {predict}")
    res = get_response(predict, intents)
    print("Your Symptom was:", txt)
    print("Result :", res)

if __name__ == '__main__':
    print("Bot is Running")
    while True:
        text = input("You may tell me your symptoms now: ")
        if text.lower() in ['exit', 'quit']:
            print("Thank you. Shutting down now.")
            break
        calling_the_bot(text)
