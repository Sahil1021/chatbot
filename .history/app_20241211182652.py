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
            background-color: #f0f2f6;
            color: #333333;
        }
        .chat-container {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            font-size: 16px;
        }
        .user-message {
            background-color: #dff9fb;
            text-align: left;
        }
        .chatbot-message {
            background-color: #f6e58d;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 style='text-align: center;'>🤖 Chatbot with NLP</h1>", unsafe_allow_html=True)

    # Sidebar with options
    st.sidebar.title("Options")
    menu = ["Home", "Conversation History", "About", "Resources"]
    choice = st.sidebar.radio("Navigate", menu)

    if choice == "Home":
        st.write("Welcome! Ask me anything, and I'll try my best to respond. 🤗")

        # Create chat log if not exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("Type your message:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)

            # Display styled chat
            st.markdown(f"<div class='chat-container user-message'>👤 <b>You:</b> {user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-container chatbot-message'>🤖 <b>Chatbot:</b> {response}</div>", unsafe_allow_html=True)

            # Save to chat log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Feedback section
            st.write("Did you find this response helpful?")
            if st.button("👍"):
                st.success("Thanks for your feedback!")
            if st.button("👎"):
                st.error("Sorry about that! We'll improve.")

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.markdown(f"👤 **User:** {row[0]}")
                    st.markdown(f"🤖 **Chatbot:** {row[1]}")
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
        st.subheader("Roadmap:")
        st.write("- Add support for multiple languages.")
        st.write("- Enable voice interaction.")
        st.write("- Integration with external APIs for dynamic responses.")

    elif choice == "Resources":
        st.header("Resources")
        st.write("- [Streamlit Documentation](https://docs.streamlit.io)")
        st.write("- [NLP with Python](https://www.nltk.org/book/)")
        st.write("- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)")

if __name__ == '__main__':
    main()
