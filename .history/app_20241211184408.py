import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

    # Add a custom background and font styling
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #89f7fe, #66a6ff); /* Gradient background */
            color: #333333;
        }
        .stTextInput > div > label {
            font-size: 18px;
        }
        .chat-container {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            font-size: 16px;
        }
        .user-message {
            background-color: #dff9fb;
            color: #000000; /* Black text for user message */
            text-align: left;
        }
        .chatbot-message {
            background-color: #f6e58d;
            color: #000000; /* Black text for chatbot message */
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖 Chatbot with NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome! Ask me anything, and I'll try my best to respond.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)

            # Display messages with user and chatbot avatars
            st.markdown(f"<div class='chat-container user-message'><strong>👤 You:</strong> {user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-container chatbot-message'><strong>🤖 Chatbot:</strong> {response}</div>", unsafe_allow_html=True)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(f"<div class='chat-container user-message'><strong>👤 User:</strong> {row[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-container chatbot-message'><strong>🤖 Chatbot:</strong> {row[1]}</div>", unsafe_allow_html=True)
                    st.markdown(f"_Timestamp:_ {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write("This chatbot demonstrates basic NLP functionality using intents and responses.")
        st.subheader("Features:")
        st.write("- Interactive conversation with intent recognition.")
        st.write("- Save conversation history.")
        st.write("- Enhanced UI with Streamlit.")

if __name__ == '__main__':
    main()
