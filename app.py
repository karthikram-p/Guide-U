import streamlit as st
st.markdown(
    """
    <style>
    .glass-title {
        font-size: 2.8em;
        font-weight: 800;
        color: #fff;
        text-align: center;
        margin-bottom: 18px;
        letter-spacing: 2px;
        text-shadow: 0 0 6px #fff, 0 2px 4px #fff8;
        background: rgba(30, 30, 30, 0.25);
        border-radius: 18px;
        box-shadow: 0 4px 12px 0 rgba(255,255,255,0.09);
        backdrop-filter: blur(8px) saturate(180%);
        -webkit-backdrop-filter: blur(8px) saturate(180%);
        border: 1.5px solid rgba(255, 255, 255, 0.09);
        padding: 18px 0 12px 0;
        transition: box-shadow 0.3s, border 0.3s;
    }
    .glass-title:hover {
        box-shadow: 0 0 12px 3px #fff, 0 4px 12px 0 rgba(255,255,255,0.09);
        border: 1.5px solid #fff;
    }
    .glass-card {
        background: rgba(30, 30, 30, 0.32);
        border-radius: 16px;
        box-shadow: 0 2px 8px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(6px) saturate(160%);
        -webkit-backdrop-filter: blur(6px) saturate(160%);
        border: 1.2px solid rgba(255, 255, 255, 0.06);
        color: #fff;
        margin-bottom: 18px;
        padding: 18px 20px 14px 20px;
        transition: box-shadow 0.3s, border 0.3s;
    }
    .glass-card:hover {
        box-shadow: 0 0 8px 2px #fff, 0 2px 8px 0 rgba(255,255,255,0.06);
        border: 1.2px solid #fff;
    }
    .glass-quiz {
        background: rgba(30, 30, 30, 0.32);
        border-radius: 16px;
        box-shadow: 0 2px 8px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(6px) saturate(160%);
        -webkit-backdrop-filter: blur(6px) saturate(160%);
        border: 1.2px solid rgba(255, 255, 255, 0.06);
        color: #fff;
        margin-bottom: 18px;
        padding: 18px 20px 14px 20px;
        transition: box-shadow 0.3s, border 0.3s;
    }
    .glass-quiz:hover {
        box-shadow: 0 0 8px 2px #fff, 0 2px 8px 0 rgba(255,255,255,0.06);
        border: 1.2px solid #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import webbrowser
import os
import google.generativeai as genai
import textwrap
from dotenv import load_dotenv, find_dotenv

try:
    from transformers import pipeline
except ImportError:
    st.error("The 'pipeline' module could not be imported from 'transformers'. Please ensure the 'transformers' library is installed and up-to-date.")
    st.stop()

# Initialize the NLP pipeline for sentiment analysis
nlp = pipeline("sentiment-analysis")

# Function to recommend academic resources (stub function)
def recommend_academic_resources(keyword, num_recommendations):
    # Stub function - replace with actual recommendation logic
    return [f"Resource {i+1} for {keyword}" for i in range(num_recommendations)]

# Function to display course details (stub function)
def display_course_details(recommendations, title):
    st.subheader(title)
    for recommendation in recommendations:
        st.write(recommendation)

# Configure API key
load_dotenv(find_dotenv(), override=True)
genai.configure(api_key='AIzaSyBCvarrNfobuZnKOXSTsVDqis1d8x3tc2s')


# Load the GenerativeModel for text (Gemini 2.0 Flash-Lite)
text_model = genai.GenerativeModel('gemini-1.5-flash-latest')

@st.cache_data
def load_datasets():
    courses_df = pd.read_csv('Online_Courses.csv')
    courses_df.drop_duplicates(subset=['Title'], inplace=True)
    courses_df.reset_index(drop=True, inplace=True)
    udemy_df = pd.read_csv('udemy_courses.csv')
    udemy_df.drop_duplicates(subset=['course_title'], inplace=True)
    udemy_df.reset_index(drop=True, inplace=True)
    excel_file = 'Metadata.xlsx'
    booklist_df = pd.read_excel(excel_file, sheet_name='Booklist')
    # Preprocess the data by dropping duplicates based on Title
    booklist_df.drop_duplicates(subset=['Book Title'], inplace=True)
    booklist_df.reset_index(drop=True, inplace=True)
    return courses_df, udemy_df, booklist_df  # Removed user_course_matrix

courses_df, udemy_df, booklist_df = load_datasets()

# Preprocess topics for FutureLearn
def preprocess_topics(topics_str):
    if pd.isna(topics_str) or topics_str is np.nan:
        return ''
    else:
        topics_str = ' '.join(topics_str.strip().split())
        topics_list = re.split(r'/', topics_str)
        unique_topics = list(set(topics_list))
        return ' '.join(unique_topics)

# Base directory where PDFs are stored
base_pdf_path = r'C:\Users\karth\Downloads\Projects\MP\SGC\data\data'

# Function to preprocess the Subject Classification column
def preprocess_subject_classification(subject_classification):
    return subject_classification.replace(';', ' ')

# Apply preprocessing to the Subject Classification column
booklist_df['Subject Classification'] = booklist_df['Subject Classification'].apply(preprocess_subject_classification)

# Define a function to apply basic formatting to the text
def simple_format(text):
    """
    Applies basic formatting to the text, like indentation and bullet points.
    """
    lines = text.strip().splitlines()  # Split into lines
    formatted_text = ""
    for line in lines:
        if line.startswith("*"):  # Bullet point
            formatted_text += "  * " + line.strip("*") + "\n"
        else:
            formatted_text += "  " + line + "\n"  # Indent non-bullet points
    return formatted_text.rstrip()  # Remove trailing newline

# Recommendation functions
def recommend_coursera(input_word, n_recommendations=15):
    courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Skills'] + ' ' + courses_df['Category'] + ' ' + courses_df['Sub-Category']
    courses_df['combined_text'] = courses_df['combined_text'].fillna('')
    coursera_courses = courses_df[courses_df['Site'] == 'Coursera']
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(coursera_courses['combined_text'])
    query_tfidf = tfidf.transform([input_word])
    content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    sorted_indices = (-content_scores).argsort()[:n_recommendations]
    recommendations = coursera_courses.iloc[sorted_indices][['Title', 'URL', 'Rating', 'Number of viewers', 'Skills']]
    recommendations['Site'] = 'Coursera'
    return recommendations.reset_index(drop=True)

def recommend_udacity(input_word, n_recommendations=7):
    courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Short Intro'] + ' ' + courses_df['Prequisites'] + ' ' + courses_df['What you learn']
    courses_df['combined_text'] = courses_df['combined_text'].fillna('')
    udacity_courses = courses_df[courses_df['Site'] == 'Udacity']
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(udacity_courses['combined_text'])
    query_tfidf = tfidf.transform([input_word])
    content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    sorted_indices = (-content_scores).argsort()[:n_recommendations]
    recommendations = udacity_courses.iloc[sorted_indices][['Title', 'URL', 'Program Type', 'Level', 'School']]
    recommendations['Site'] = 'Udacity'
    return recommendations.reset_index(drop=True)

def recommend_futurelearn(input_word, n_recommendations=7):
    courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Topics related to CRM'] 
    courses_df['combined_text'] = courses_df['combined_text'].fillna('')
    courses_df['Topics related to CRM'] = courses_df['Topics related to CRM'].apply(preprocess_topics)
    courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Topics related to CRM'] + ' ' + courses_df['Course Short Intro']
    futurelearn_courses = courses_df[courses_df['Site'] == 'Future Learn']
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(futurelearn_courses['combined_text'])
    query_tfidf = tfidf.transform([input_word])
    content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    sorted_indices = (-content_scores).argsort()[:n_recommendations]
    recommendations = futurelearn_courses.iloc[sorted_indices][['Title', 'URL']]
    recommendations['Site'] = 'FutureLearn'
    return recommendations.reset_index(drop=True)

def recommend_simplilearn(input_word, n_recommendations=7):
    courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Short Intro'] + ' ' + courses_df['COURSE CATEGORIES']
    courses_df['combined_text'] = courses_df['combined_text'].fillna('')
    simplilearn_courses = courses_df[courses_df['Site'] == 'Simplilearn']
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(simplilearn_courses['combined_text'])
    query_tfidf = tfidf.transform([input_word])
    content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    sorted_indices = (-content_scores).argsort()[:n_recommendations]
    recommendations = simplilearn_courses.iloc[sorted_indices][['Title', 'URL', 'Number of ratings']]
    recommendations['Site'] = 'SimpliLearn'
    return recommendations.reset_index(drop=True)

def recommend_udemy(input_word, n_recommendations=15):
    udemy_df['combined_text'] = udemy_df['course_title'] + ' ' + udemy_df['subject']
    udemy_df['combined_text'] = udemy_df['combined_text'].fillna('')
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(udemy_df['combined_text'])
    query_tfidf = tfidf.transform([input_word])
    content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    sorted_indices = (-content_scores).argsort()[:n_recommendations]
    recommendations = udemy_df.iloc[sorted_indices][['course_title', 'url', 'is_paid', 'price', 'num_subscribers', 'num_reviews', 'num_lectures', 'level', 'content_duration']]
    return recommendations.reset_index(drop=True)

def get_related_courses(keyword, top_n=4):
    # Find all courses that contain the keyword
    keyword_courses = [course for course in courses_df['Title'] if keyword.lower() in course.lower()]

    # Filter courses to include only those containing the keyword
    filtered_courses = courses_df[courses_df['Title'].str.contains(keyword, case=False, na=False)]

    # Get the top recommended courses
    top_n_courses = filtered_courses.head(top_n)['Title'].tolist()
    return top_n_courses

# Function to display course details
def display_course_details(recommendations, site):
    if site == 'Coursera':
        st.write("Coursera Recommendations:")
        for index, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="margin-bottom: 20px;">
                    <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                    <p><strong>Rating:</strong> {row.get('Rating', 'N/A')}</p>
                    <p><strong>Number of Students Enrolled:</strong> {row.get('Number of viewers', 'N/A')}</p>
                    <p><strong>Skills:</strong> {row.get('Skills', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True
            )
    # ... (rest of the display_course_details function)

option = st.sidebar.selectbox(
    'Choose the platform or resource type for recommendations:',
    ('Home', 'Quiz Section', 'Skill Development Courses', 'Recommendations By Gemini', 'Feedback')
)

if option == 'Home':
    # Updated Home Page
    st.title("Welcome to Guide-U")
    st.markdown(
        """
        <style>
        .glass-box {
            background: rgba(30, 30, 30, 0.35);
            border-radius: 18px;
            box-shadow: 0 4px 24px 0 rgba(255,255,255,0.18);
            backdrop-filter: blur(8px) saturate(180%);
            -webkit-backdrop-filter: blur(8px) saturate(180%);
            border: 1.5px solid rgba(255, 255, 255, 0.18);
            padding: 32px 28px 28px 28px;
            margin-bottom: 32px;
            color: #fff;
            transition: box-shadow 0.3s, border 0.3s;
        }
        .glass-box:hover {
            box-shadow: 0 0 24px 6px #fff, 0 4px 24px 0 rgba(255,255,255,0.18);
            border: 1.5px solid #fff;
        }
        .glass-box h3 {
            color: #fff;
            margin-top: 0;
            margin-bottom: 18px;
            letter-spacing: 1px;
            font-weight: 700;
            text-shadow: 0 2px 8px #fff8;
        }
        .glass-box ul {
            font-size: 1.13em;
            margin-bottom: 18px;
        }
        .glass-box li {
            margin-bottom: 10px;
            line-height: 1.6;
        }
        .glass-box p {
            color: #fff;
            font-weight: 500;
            text-shadow: 0 0 8px #fff8;
        }
        </style>
        <div class="glass-box">
            <h3>App Features</h3>
            <ul>
                <li><b>Skill Development Courses:</b> Get personalized course recommendations from Coursera, Udacity, FutureLearn, SimpliLearn, and Udemy based on your interests.</li>
                <li><b>Recommendations By Gemini:</b> Leverage AI to receive tailored learning and career suggestions for any topic or skill.</li>
                <li><b>Quiz Section:</b> Take domain-specific quizzes and receive a detailed career roadmap based on your performance.</li>
                <li><b>Feedback:</b> Share your feedback or suggestions to help us improve the app.</li>
            </ul>
            <p>Use the sidebar to navigate between features and make the most of your learning journey!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif option == 'Skill Development Courses':
    st.header("Skill Development Recommendations")

    # Input for course name
    input_word = st.text_input("Enter a course name or keyword:", value="")

    # Single slider for selecting the number of recommendations for all platforms
    num_recommendations = st.slider('Select the number of recommendations per platform:', 1, 30, 10)

    if input_word:
        # Fetch recommendations from all platforms
        coursera_recommendations = recommend_coursera(input_word, num_recommendations)
        udacity_recommendations = recommend_udacity(input_word, num_recommendations)
        futurelearn_recommendations = recommend_futurelearn(input_word, num_recommendations)
        simplilearn_recommendations = recommend_simplilearn(input_word, num_recommendations)
        udemy_recommendations = recommend_udemy(input_word, num_recommendations)

        # Display recommendations for each platform in separate scrollable sections styled as cue cards
        st.subheader("Recommendations from Coursera:")
        with st.container():
            for index, row in coursera_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                        <p><strong>Rating:</strong> {row.get('Rating', 'N/A')}</p>
                        <p><strong>Number of Students Enrolled:</strong> {row.get('Number of viewers', 'N/A')}</p>
                        <p><strong>Skills:</strong> {row.get('Skills', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

        st.subheader("Recommendations from Udacity:")
        with st.container():
            for index, row in udacity_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                        <p><strong>Program Type:</strong> {row.get('Program Type', 'N/A')}</p>
                        <p><strong>Level:</strong> {row.get('Level', 'N/A')}</p>
                        <p><strong>School:</strong> {row.get('School', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

        st.subheader("Recommendations from FutureLearn:")
        with st.container():
            for index, row in futurelearn_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                    </div>
                    """, unsafe_allow_html=True
                )

        st.subheader("Recommendations from SimpliLearn:")
        with st.container():
            for index, row in simplilearn_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                        <p><strong>Number of Ratings:</strong> {row.get('Number of ratings', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

        st.subheader("Recommendations from Udemy:")
        with st.container():
            for index, row in udemy_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4><a href="{row['url']}" target="_blank" style="text-decoration:none;">{row['course_title']}</a></h4>
                        <p><strong>Is Paid:</strong> {row.get('is_paid', 'N/A')}</p>
                        <p><strong>Price:</strong> {row.get('price', 'N/A')}</p>
                        <p><strong>Number of Subscribers:</strong> {row.get('num_subscribers', 'N/A')}</p>
                        <p><strong>Number of Reviews:</strong> {row.get('num_reviews', 'N/A')}</p>
                        <p><strong>Number of Lectures:</strong> {row.get('num_lectures', 'N/A')}</p>
                        <p><strong>Level:</strong> {row.get('level', 'N/A')}</p>
                        <p><strong>Content Duration:</strong> {row.get('content_duration', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

elif option == 'Recommendations By Gemini':
    st.header("Recommendations By Gemini")

    input_word = st.text_input("Enter a topic or skill for recommendations:", value="")

    if st.button("Generate Recommendations"):
        with st.spinner("Fetching recommendations from Gemini..."):
            response_text = text_model.generate_content(
                f"Provide a detailed roadmap consisting of resources, including YouTube channels, online courses, and roadmaps for '{input_word}' in 2025 required for a engineering student in Telangana. Include links and categorize them by beginner, intermediate, and advanced levels. Be Precise and concise. Use bullet points for clarity. Also, provide a brief overview of the topic and its relevance in 2025."
            )
        st.subheader("AI-Generated Recommendations:")
        st.write(simple_format(response_text.text))

elif option == 'Feedback':
    st.title("Feedback")

    feedback_text = st.text_area("Share your feedback or suggestions:", height=200)

    if st.button("Submit Feedback"):
        # Analyze sentiment of the feedback
        sentiment = nlp(feedback_text)[0]['label']

        # Store feedback in session state (in-memory, suitable for demo/deployment)
        if 'feedback_list' not in st.session_state:
            st.session_state['feedback_list'] = []
        st.session_state['feedback_list'].append({'sentiment': sentiment, 'feedback': feedback_text})

        # Show custom message based on sentiment
        if sentiment.lower() == 'positive':
            st.success("Thank you for your positive feedback! We're glad you enjoyed the app.")
        elif sentiment.lower() == 'negative':
            st.warning("Thank you for your feedback. We will work to improve the app based on your suggestions.")
        else:
            st.info("Thank you for your feedback!")

    # Optionally, show all feedbacks to the user (for demo/testing)
    if 'feedback_list' in st.session_state and st.session_state['feedback_list']:
        with st.expander("See all submitted feedback (this session only)"):
            for fb in st.session_state['feedback_list']:
                st.write(f"Sentiment: {fb['sentiment']} | Feedback: {fb['feedback']}")

elif option == 'Quiz Section':
    st.markdown('<div class="glass-title">Quiz Section</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-quiz">Select a domain, take a quiz, and receive a career roadmap based on your performance!</div>', unsafe_allow_html=True)

    # List of domains
    domains = [
        "Artificial Intelligence",
        "Data Science",
        "Web Development",
        "Cybersecurity",
        "Cloud Computing",
        "Embedded Systems",
        "Blockchain",
        "Game Development",
        "Robotics",
        "IoT (Internet of Things)"
    ]

    # Select a domain
    selected_domain = st.selectbox("Choose your domain of interest:", domains)

    # Quiz questions for each domain (added placeholders for missing domains)
    quiz_questions = {
        "Artificial Intelligence": {
            "basic": [
                {"question": "What is the primary goal of AI?", "options": ["Mimic human intelligence", "Solve math problems", "Build robots", "Create websites"], "answer": "Mimic human intelligence"},
                {"question": "Which algorithm is used for supervised learning?", "options": ["K-Means", "Linear Regression", "Apriori", "PCA"], "answer": "Linear Regression"},
                {"question": "What does ML stand for in AI?", "options": ["Machine Learning", "Multiple Layers", "Meta Language", "Module Links"], "answer": "Machine Learning"},
                {"question": "Which of these is NOT a type of AI?", "options": ["ANI", "AGI", "ASI", "AEI"], "answer": "AEI"},
                {"question": "Which programming language is most commonly used for AI development?", "options": ["Java", "C++", "Python", "Ruby"], "answer": "Python"},
                {"question": "What is a decision tree?", "options": ["A visualization tool", "A machine learning algorithm", "A hardware component", "A programming framework"], "answer": "A machine learning algorithm"},
                {"question": "What does the Turing Test evaluate?", "options": ["Computer processing speed", "Computer memory capacity", "Computer's ability to exhibit human-like behavior", "Computer graphics capability"], "answer": "Computer's ability to exhibit human-like behavior"},
                {"question": "Which of these is an example of unsupervised learning?", "options": ["Linear regression", "Decision trees", "Clustering", "Support vector machines"], "answer": "Clustering"},
                {"question": "What is an epoch in machine learning?", "options": ["A type of neural network", "One complete pass through the training dataset", "A validation technique", "A performance metric"], "answer": "One complete pass through the training dataset"},
                {"question": "What is feature extraction?", "options": ["Selecting important variables from data", "Creating new features from existing ones", "Removing outliers from data", "Normalizing data values"], "answer": "Creating new features from existing ones"},
                {"question": "What is AI bias?", "options": ["When AI favors certain programming languages", "When AI produces unfair outcomes for certain groups", "When AI consumes too much computing power", "When AI models are too complex"], "answer": "When AI produces unfair outcomes for certain groups"},
                {"question": "What is an expert system?", "options": ["A system that uses AI to mimic human experts", "A system built by expert programmers", "A fast computing system", "A system that trains AI researchers"], "answer": "A system that uses AI to mimic human experts"},
                {"question": "Which field combines AI with statistics?", "options": ["Computational Linguistics", "Data Science", "Computer Vision", "Robotics"], "answer": "Data Science"},
                {"question": "What is classification in machine learning?", "options": ["Organizing data into folders", "Predicting a category for input data", "Clustering similar data points", "Reducing dimensionality"], "answer": "Predicting a category for input data"},
                {"question": "What is natural language processing?", "options": ["Programming in simple English", "AI understanding and generating human language", "Text translation software", "Voice recognition only"], "answer": "AI understanding and generating human language"}
            ],
            "medium": [
                {"question": "What is a neural network?", "options": ["A type of database", "A machine learning model", "A programming language", "A hardware device"], "answer": "A machine learning model"},
                {"question": "What is reinforcement learning?", "options": ["Learning from labeled data", "Learning from rewards and penalties", "Learning from unlabeled data", "Learning from examples"], "answer": "Learning from rewards and penalties"},
                {"question": "What does backpropagation do in neural networks?", "options": ["Updates weights based on error", "Initializes random weights", "Processes input data", "Stores model parameters"], "answer": "Updates weights based on error"},
                {"question": "What is transfer learning?", "options": ["Transferring data between computers", "Using knowledge from one model to another", "Converting models to different formats", "Moving AI systems between platforms"], "answer": "Using knowledge from one model to another"},
                {"question": "What is an activation function in neural networks?", "options": ["A function that starts the learning process", "A function that determines the output of a node", "A function that ends training", "A function that measures error"], "answer": "A function that determines the output of a node"},
                {"question": "What is the purpose of dropout in neural networks?", "options": ["To speed up training", "To prevent overfitting", "To initialize weights", "To improve accuracy"], "answer": "To prevent overfitting"},
                {"question": "What is ensemble learning?", "options": ["Training multiple models together", "Combining predictions from multiple models", "Learning in groups", "Sequential model training"], "answer": "Combining predictions from multiple models"},
                {"question": "What is a GAN?", "options": ["Generative Adversarial Network", "Global AI Network", "General Approximation Node", "Gradient Analysis Network"], "answer": "Generative Adversarial Network"},
                {"question": "What is regularization in machine learning?", "options": ["Making rules for data formatting", "Techniques to prevent overfitting", "Normalizing input values", "Organizing training data"], "answer": "Techniques to prevent overfitting"},
                {"question": "What are hyperparameters?", "options": ["Parameters set before training", "Parameters learned during training", "Parameters used for evaluation", "Parameters for deployment"], "answer": "Parameters set before training"},
                {"question": "What is a convolutional neural network used for?", "options": ["Natural language processing", "Image recognition", "Audio processing", "Time series analysis"], "answer": "Image recognition"},
                {"question": "What is a recurrent neural network?", "options": ["Networks that process sequences", "Networks with only one layer", "Networks without hidden layers", "Networks that only use binary inputs"], "answer": "Networks that process sequences"},
                {"question": "What is semi-supervised learning?", "options": ["Learning with partially labeled data", "Learning with small datasets", "Learning with human supervision", "Learning with multiple algorithms"], "answer": "Learning with partially labeled data"},
                {"question": "What does LSTM stand for?", "options": ["Large Scale Training Model", "Long Short-Term Memory", "Linear Statistical Training Method", "Lightweight System Training Module"], "answer": "Long Short-Term Memory"},
                {"question": "What is the vanishing gradient problem?", "options": ["When gradients become too small during training", "When models take too long to train", "When input data varies too much", "When neural networks have too many layers"], "answer": "When gradients become too small during training"}
            ],
            "advanced": [
                {"question": "What is the vanishing gradient problem?", "options": ["Gradients become too small during backpropagation", "Gradients become too large during backpropagation", "Model overfits the data", "Model underfits the data"], "answer": "Gradients become too small during backpropagation"},
                {"question": "Which activation function is commonly used in deep learning?", "options": ["ReLU", "Sigmoid", "Tanh", "Softmax"], "answer": "ReLU"},
                {"question": "What is attention mechanism in deep learning?", "options": ["A way to focus on important parts of input", "A technique to keep neural networks alert", "A method to visualize neural networks", "A process to validate model outputs"], "answer": "A way to focus on important parts of input"},
                {"question": "What is a transformer model?", "options": ["A model that converts data formats", "A neural network architecture using self-attention", "A model that transforms images", "A hardware component for AI"], "answer": "A neural network architecture using self-attention"},
                {"question": "What is explainable AI?", "options": ["AI that can explain concepts to humans", "AI with transparent decision-making processes", "AI that generates explanations", "AI developed with clear documentation"], "answer": "AI with transparent decision-making processes"},
                {"question": "What is federated learning?", "options": ["Learning across multiple institutions", "Training models on decentralized data", "A centralized learning approach", "An international AI standard"], "answer": "Training models on decentralized data"},
                {"question": "What is a variational autoencoder?", "options": ["A generative model with probabilistic encodings", "A validation framework", "A type of RNN", "An optimization algorithm"], "answer": "A generative model with probabilistic encodings"},
                {"question": "What is meta-learning?", "options": ["Learning about learning processes", "Training models to learn quickly", "Learning from metadata", "Training on multiple datasets"], "answer": "Training models to learn quickly"},
                {"question": "What is Q-learning in reinforcement learning?", "options": ["A model-free algorithm", "A supervised learning technique", "A quality assurance process", "A data preparation method"], "answer": "A model-free algorithm"},
                {"question": "What is the lottery ticket hypothesis?", "options": ["A theory about random weight initialization", "A theory about finding sparse trainable subnetworks", "A method for distributing AI profits", "A training optimization technique"], "answer": "A theory about finding sparse trainable subnetworks"},
                {"question": "What are adversarial examples?", "options": ["Examples that contradict each other", "Inputs designed to fool AI models", "Examples of model failures", "Training data that causes conflicts"], "answer": "Inputs designed to fool AI models"},
                {"question": "What is self-supervised learning?", "options": ["AI creating its own training data", "Learning without any supervision", "Models evaluating their own performance", "Models learning from unlabeled data using derived labels"], "answer": "Models learning from unlabeled data using derived labels"},
                {"question": "What is few-shot learning?", "options": ["Learning from very few examples", "Learning in short time periods", "Learning with minimal computing resources", "Learning simple concepts"], "answer": "Learning from very few examples"},
                {"question": "What is a diffusion model?", "options": ["A model that spreads information", "A generative model based on denoising", "A model for fluid dynamics", "A model that distributes computation"], "answer": "A generative model based on denoising"},
                {"question": "What are foundation models in AI?", "options": ["Basic models for beginners", "Large models trained on broad data that can be adapted", "Models that form the foundation of AI theory", "Models with guaranteed performance"], "answer": "Large models trained on broad data that can be adapted"}
            ]
        },
        "Data Science": {
            "basic": [
                {"question": "What is the first step in data analysis?", "options": ["Data Cleaning", "Data Visualization", "Data Collection", "Model Building"], "answer": "Data Collection"},
                {"question": "Which library is commonly used for data manipulation in Python?", "options": ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"], "answer": "Pandas"},
                {"question": "What is a dataset?", "options": ["A collection of data", "A database management system", "A programming language", "A visualization tool"], "answer": "A collection of data"},
                {"question": "What does EDA stand for in data science?", "options": ["Electronic Data Analysis", "Extended Data Architecture", "Exploratory Data Analysis", "External Data Acquisition"], "answer": "Exploratory Data Analysis"},
                {"question": "What is a histogram used for?", "options": ["Showing relationships between variables", "Displaying the distribution of data", "Tracking data over time", "Comparing categories"], "answer": "Displaying the distribution of data"},
                {"question": "What is data cleaning?", "options": ["Removing all data", "Handling missing values and errors", "Encrypting sensitive data", "Compressing data files"], "answer": "Handling missing values and errors"},
                {"question": "What is a correlation coefficient?", "options": ["A measure of central tendency", "A measure of relationship between variables", "A measure of data spread", "A data visualization technique"], "answer": "A measure of relationship between variables"},
                {"question": "What is descriptive statistics?", "options": ["Predicting future trends", "Summarizing and describing data features", "Testing hypotheses", "Drawing inferential conclusions"], "answer": "Summarizing and describing data features"},
                {"question": "What is a bar chart best used for?", "options": ["Showing continuous data changes", "Comparing different categories", "Showing distribution of data", "Displaying correlations"], "answer": "Comparing different categories"},
                {"question": "What does SQL stand for?", "options": ["Structured Query Language", "Simple Query Language", "System Quality Language", "Sequential Query Logic"], "answer": "Structured Query Language"},
                {"question": "What is an outlier?", "options": ["A data point far from others", "A prediction error", "A missing value", "A duplicate record"], "answer": "A data point far from others"},
                {"question": "What is data normalization?", "options": ["Creating backup data", "Scaling features to a common range", "Removing duplicate data", "Converting data to text format"], "answer": "Scaling features to a common range"},
                {"question": "Which measure is affected most by outliers?", "options": ["Mean", "Median", "Mode", "Range"], "answer": "Mean"},
                {"question": "What is a dashboard in data science?", "options": ["A control panel for computers", "A visual display of key metrics", "A database management tool", "A programming interface"], "answer": "A visual display of key metrics"},
                {"question": "What is the purpose of data visualization?", "options": ["To make data look attractive", "To communicate information clearly", "To store data efficiently", "To encrypt sensitive information"], "answer": "To communicate information clearly"}
            ],
            "medium": [
                {"question": "What does 'overfitting' mean in machine learning?", "options": ["Model performs well on training data but poorly on test data", "Model performs poorly on all data", "Model performs well on all data", "Model is too simple"], "answer": "Model performs well on training data but poorly on test data"},
                {"question": "Which metric is used to evaluate classification models?", "options": ["Accuracy", "Mean Squared Error", "R-squared", "Silhouette Score"], "answer": "Accuracy"},
                {"question": "What is feature engineering?", "options": ["Creating new features from existing data", "Selecting the best model", "Tuning model parameters", "Cleaning raw data"], "answer": "Creating new features from existing data"},
                {"question": "What is a confusion matrix?", "options": ["A table showing model prediction errors", "A visualization of correlations", "A matrix of training data", "A complex mathematical formula"], "answer": "A table showing model prediction errors"},
                {"question": "What is K-fold cross-validation?", "options": ["Dividing data into K equal parts for validation", "Using K different models for prediction", "Reducing dimensionality to K components", "Training a model K times"], "answer": "Dividing data into K equal parts for validation"},
                {"question": "What is regularization in machine learning?", "options": ["Making model training regular", "Adding constraints to reduce overfitting", "Creating regular intervals in data", "Standardizing input data"], "answer": "Adding constraints to reduce overfitting"},
                {"question": "What is data wrangling?", "options": ["Visualizing data", "Storing data securely", "Cleaning and transforming raw data", "Analyzing data for insights"], "answer": "Cleaning and transforming raw data"},
                {"question": "What is a time series?", "options": ["Data collected at specific time intervals", "A type of database", "A visualization technique", "A forecasting method"], "answer": "Data collected at specific time intervals"},
                {"question": "What is a box plot used for?", "options": ["Showing exact values", "Displaying distribution and outliers", "Comparing only two variables", "Tracking changes over time"], "answer": "Displaying distribution and outliers"},
                {"question": "What is the purpose of dimensionality reduction?", "options": ["To make data storage easier", "To visualize data in fewer dimensions", "To reduce computation time and complexity", "All of the above"], "answer": "All of the above"},
                {"question": "What is an ROC curve used for?", "options": ["Visualizing regression performance", "Evaluating classification model performance", "Showing data correlations", "Tracking time series"], "answer": "Evaluating classification model performance"},
                {"question": "What is the difference between supervised and unsupervised learning?", "options": ["Speed of execution", "Amount of data required", "Presence of labeled data", "Programming language used"], "answer": "Presence of labeled data"},
                {"question": "What is A/B testing?", "options": ["Testing two versions of something to see which performs better", "Testing a hypothesis against a baseline", "Testing models against each other", "Testing various feature combinations"], "answer": "Testing two versions of something to see which performs better"},
                {"question": "What is ensemble learning?", "options": ["Learning in groups", "Combining multiple models", "Learning from multiple datasets", "A visualization technique"], "answer": "Combining multiple models"},
                {"question": "What is the purpose of data imputation?", "options": ["Adding new data points", "Filling in missing values", "Removing outliers", "Scaling features"], "answer": "Filling in missing values"}
            ],
            "advanced": [
                {"question": "What is the purpose of cross-validation?", "options": ["To split data into training and test sets", "To evaluate model performance on unseen data", "To optimize hyperparameters", "To reduce overfitting"], "answer": "To evaluate model performance on unseen data"},
                {"question": "What is PCA used for?", "options": ["Dimensionality reduction", "Classification", "Clustering", "Regression"], "answer": "Dimensionality reduction"},
                {"question": "What is deep learning?", "options": ["Learning from very large datasets", "Neural networks with many layers", "Learning complex relationships", "All of the above"], "answer": "Neural networks with many layers"},
                {"question": "What is the bias-variance tradeoff?", "options": ["A tradeoff between model complexity and accuracy", "A tradeoff between training time and model size", "A tradeoff between underfitting and overfitting", "A tradeoff between interpretability and performance"], "answer": "A tradeoff between underfitting and overfitting"},
                {"question": "What is bagging in ensemble learning?", "options": ["Bootstrapping data and aggregating results", "Combining weak models sequentially", "Weighing models based on performance", "Selecting the best model from many"], "answer": "Bootstrapping data and aggregating results"},
                {"question": "What is gradient boosting?", "options": ["A visualization technique", "An optimization algorithm", "An ensemble technique building models sequentially", "A method for feature selection"], "answer": "An ensemble technique building models sequentially"},
                {"question": "What is LSTM used for?", "options": ["Processing sequential data", "Image recognition", "Dimensionality reduction", "Feature selection"], "answer": "Processing sequential data"},
                {"question": "What is a Bayesian network?", "options": ["A neural network variant", "A probabilistic graphical model", "A database system", "A clustering algorithm"], "answer": "A probabilistic graphical model"},
                {"question": "What is anomaly detection?", "options": ["Finding unusual patterns in data", "Detecting data quality issues", "Identifying outliers", "All of the above"], "answer": "Finding unusual patterns in data"},
                {"question": "What is a recommender system?", "options": ["A system that recommends models to use", "A system that predicts user preferences", "A system for data cleaning recommendations", "A system that optimizes hyperparameters"], "answer": "A system that predicts user preferences"},
                {"question": "What does SHAP stand for in model explainability?", "options": ["System for Hyper-Accurate Predictions", "SHapley Additive exPlanations", "Statistical Hypothesis And Predictions", "Systematic Hierarchical Analysis Process"], "answer": "SHapley Additive exPlanations"},
                {"question": "What is transfer learning?", "options": ["Transferring data between systems", "Using knowledge from one model on a different task", "Learning how to transfer files", "Moving models between platforms"], "answer": "Using knowledge from one model on a different task"},
                {"question": "What is the curse of dimensionality?", "options": ["Having too many features for effective analysis", "When algorithms are too complex", "When data storage becomes expensive", "When visualization becomes impossible"], "answer": "Having too many features for effective analysis"},
                {"question": "What is semi-supervised learning?", "options": ["Learning with a mix of labeled and unlabeled data", "Learning with limited computational resources", "Learning with both human and AI supervision", "Learning with partial features"], "answer": "Learning with a mix of labeled and unlabeled data"},
                {"question": "What is sparsity in machine learning?", "options": ["Having few non-zero elements", "Having limited computational resources", "Having limited training data", "Having few features"], "answer": "Having few non-zero elements"}
            ]
        },
        "Web Development": {
            "basic": [
                {"question": "What does HTML stand for?", "options": ["Hyper Text Markup Language", "High Tech Modern Language", "Hyper Technical Machine Learning", "Home Tool Markup Language"], "answer": "Hyper Text Markup Language"},
                {"question": "Which language is used for styling web pages?", "options": ["HTML", "CSS", "JavaScript", "PHP"], "answer": "CSS"},
                {"question": "What is the purpose of JavaScript in web development?", "options": ["To style web pages", "To create database structures", "To add interactivity to web pages", "To create server architecture"], "answer": "To add interactivity to web pages"},
                {"question": "Which tag is used to create a hyperlink in HTML?", "options": ["<link>", "<a>", "<href>", "<url>"], "answer": "<a>"},
                {"question": "What is the correct way to include CSS in an HTML document?", "options": ["Using <style> tags", "Using a <link> tag", "Inline with style attribute", "All of the above"], "answer": "All of the above"},
                {"question": "What are HTML attributes?", "options": ["Additional information about elements", "Styling properties", "JavaScript functions", "Database connections"], "answer": "Additional information about elements"},
                {"question": "What is a responsive website?", "options": ["A website that loads quickly", "A website that adapts to different screen sizes", "A website with animations", "A website with user authentication"], "answer": "A website that adapts to different screen sizes"},
                {"question": "What does CSS stand for?", "options": ["Cascading Style Sheets", "Computer Style System", "Creative Style Solutions", "Colorful Style Sheets"], "answer": "Cascading Style Sheets"},
                {"question": "Which HTML tag is used for creating a table?", "options": ["<table>", "<tab>", "<tr>", "<grid>"], "answer": "<table>"},
                {"question": "What is a web browser?", "options": ["A program for searching the web", "A program for displaying web pages", "A tool for creating websites", "A type of server"], "answer": "A program for displaying web pages"},
                {"question": "What is the purpose of the HTML <head> tag?", "options": ["To contain headings", "To contain metadata about the document", "To define the main content", "To create navigation"], "answer": "To contain metadata about the document"},
                {"question": "Which CSS property is used to change text color?", "options": ["text-color", "font-color", "color", "text-style"], "answer": "color"},
                {"question": "What is an HTML form used for?", "options": ["Formatting text", "Creating tables", "Collecting user input", "Displaying images"], "answer": "Collecting user input"},
                {"question": "What is the default display property for a <div> element?", "options": ["inline", "block", "inline-block", "flex"], "answer": "block"},
                {"question": "What is the role of a web server?", "options": ["To create web pages", "To store web pages", "To deliver web pages to clients", "To design websites"], "answer": "To deliver web pages to clients"}
            ],
            "medium": [
                {"question": "What is the Box Model in CSS?", "options": ["A layout system for 3D elements", "A framework for responsive design", "A model defining how elements are sized with margin, border, padding and content", "A tool for aligning elements"], "answer": "A model defining how elements are sized with margin, border, padding and content"},
                {"question": "What is AJAX?", "options": ["A programming language", "A cleaning product for websites", "A technique for asynchronous web updates", "A web browser"], "answer": "A technique for asynchronous web updates"},
                {"question": "What is the difference between GET and POST methods?", "options": ["GET is secure, POST is not", "GET requests can be bookmarked, POST cannot", "GET is faster, POST is slower", "GET is for receiving data, POST is only for forms"], "answer": "GET requests can be bookmarked, POST cannot"},
                {"question": "What is a cookie in web development?", "options": ["A snack for developers", "A small piece of data stored on the client", "A server-side script", "A type of web animation"], "answer": "A small piece of data stored on the client"},
                {"question": "What is a CSS preprocessor?", "options": ["A tool that processes CSS after it's written", "A program that converts CSS to HTML", "A script that enhances CSS with variables and functions", "A performance optimization tool"], "answer": "A script that enhances CSS with variables and functions"},
                {"question": "What is a CDN?", "options": ["Content Distribution Network", "Creative Design Navigation", "Code Development Namespace", "Custom Domain Name"], "answer": "Content Distribution Network"},
                {"question": "What is the purpose of media queries in CSS?", "options": ["To embed media files", "To query a media database", "To apply styles based on device characteristics", "To optimize media loading"], "answer": "To apply styles based on device characteristics"},
                {"question": "What is localStorage in web development?", "options": ["A server database", "Client-side storage that persists after browser close", "A folder for storing local files", "A browser cache"], "answer": "Client-side storage that persists after browser close"},
                {"question": "What is the purpose of the z-index property in CSS?", "options": ["To control the horizontal position", "To control the vertical position", "To control stacking order of elements", "To zoom elements"], "answer": "To control stacking order of elements"},
                {"question": "What is a single-page application (SPA)?", "options": ["A website with only one page", "A web app that loads a single HTML page and updates dynamically", "A minimalist website design", "A mobile-only website"], "answer": "A web app that loads a single HTML page and updates dynamically"},
                {"question": "What is the purpose of webpack?", "options": ["To create web servers", "To bundle and manage web assets", "To test web applications", "To deploy websites"], "answer": "To bundle and manage web assets"},
                {"question": "What is cross-site scripting (XSS)?", "options": ["A technique for optimizing websites", "A security vulnerability allowing injection of scripts", "A method for cross-browser compatibility", "A way to share scripts between sites"], "answer": "A security vulnerability allowing injection of scripts"},
                {"question": "What is the virtual DOM?", "options": ["A fake domain name", "An in-memory representation of the real DOM", "A virtual server for hosting", "A 3D version of a website"], "answer": "An in-memory representation of the real DOM"},
                {"question": "What is a RESTful API?", "options": ["An API that needs frequent breaks", "An API following representational state transfer principles", "A testing framework for APIs", "An API documentation standard"], "answer": "An API following representational state transfer principles"},
                {"question": "What are Web Components?", "options": ["Reusable custom elements", "Hardware components for web servers", "Visual parts of a webpage", "Components of the HTTP protocol"], "answer": "Reusable custom elements"}
            ],
            "advanced": [
                {"question": "What is server-side rendering (SSR)?", "options": ["Rendering 3D graphics on servers", "Generating HTML on the server before sending to client", "A rendering engine for servers", "Optimizing server performance"], "answer": "Generating HTML on the server before sending to client"},
                {"question": "What is CORS?", "options": ["Cross-Origin Resource Sharing", "Complex Object Rendering System", "Cascading Override Response System", "Client-Oriented Response Specification"], "answer": "Cross-Origin Resource Sharing"},
                {"question": "What is a Progressive Web App (PWA)?", "options": ["An app that loads progressively", "A modern web app with native-like capabilities", "A politically progressive app", "An app that shows user progress"], "answer": "A modern web app with native-like capabilities"},
                {"question": "What is GraphQL?", "options": ["A graph-based database", "A query language for APIs", "A graphics library for web", "A CSS grid framework"], "answer": "A query language for APIs"},
                {"question": "What is the JAMstack?", "options": ["JavaScript, APIs, and Markup architecture", "Java, Apache, MySQL stack", "Just Another Mobile stack", "JavaScript And More stack"], "answer": "JavaScript, APIs, and Markup architecture"},
                {"question": "What is WebAssembly?", "options": ["A way to write assembly directly in browsers", "A binary instruction format for stack-based virtual machines", "A JavaScript assembler", "A tool for assembling web components"], "answer": "A binary instruction format for stack-based virtual machines"},
                {"question": "What is serverless architecture?", "options": ["Websites without servers", "Building apps without managing server infrastructure", "Client-side only applications", "Peer-to-peer web hosting"], "answer": "Building apps without managing server infrastructure"},
                {"question": "What is the Shadow DOM?", "options": ["Hidden DOM elements", "A dark mode for the DOM", "An encapsulated DOM tree within an element", "A secondary rendering engine"], "answer": "An encapsulated DOM tree within an element"},
                {"question": "What is code splitting in modern web development?", "options": ["Dividing code between developers", "Breaking code into smaller files loaded on demand", "Splitting code between client and server", "Separating HTML, CSS, and JS"], "answer": "Breaking code into smaller files loaded on demand"},
                {"question": "What is tree shaking in JavaScript bundling?", "options": ["A method to organize code hierarchically", "Removing dead code from bundles", "Optimizing DOM tree operations", "A visualization technique"], "answer": "Removing dead code from bundles"},
                {"question": "What is a microservice architecture?", "options": ["Using tiny services for web functions", "An architectural style that structures an application as small independent services", "A minimalist user interface", "A compact server setup"], "answer": "An architectural style that structures an application as small independent services"},
                {"question": "What is Isomorphic/Universal JavaScript?", "options": ["JavaScript that works the same everywhere", "JavaScript that runs on both client and server", "A universal standard for JavaScript", "JavaScript with universal browser support"], "answer": "JavaScript that runs on both client and server"},
                {"question": "What is a headless CMS?", "options": ["A CMS without a user interface", "A CMS that provides content through APIs rather than a traditional frontend", "A minimalist CMS", "A CMS without admin capabilities"], "answer": "A CMS that provides content through APIs rather than a traditional frontend"},
                {"question": "What are Web Workers?", "options": ["People who build websites", "Scripts that run in background threads", "Services that maintain web servers", "Tools that automate web development"], "answer": "Scripts that run in background threads"},
                {"question": "What is an event loop in JavaScript?", "options": ["A loop that creates events", "A mechanism to handle asynchronous callbacks", "A special type of for-loop", "A user interaction tracker"], "answer": "A mechanism to handle asynchronous callbacks"}
            ]
        },
        "Cybersecurity": {
            "basic": [
                {"question": "What is a firewall?", "options": ["A physical barrier to prevent fires", "Software/hardware that monitors network traffic", "A type of computer virus", "A backup system"], "answer": "Software/hardware that monitors network traffic"},
                {"question": "What is phishing?", "options": ["A fishing technique", "Sending deceptive emails to steal information", "A network protocol", "A programming language"], "answer": "Sending deceptive emails to steal information"},
                {"question": "What is encryption?", "options": ["Compressing data", "Converting data into a code to prevent unauthorized access", "Backing up data", "Deleting sensitive data"], "answer": "Converting data into a code to prevent unauthorized access"},
                {"question": "What is malware?", "options": ["Malicious software", "Hardware failure", "Network equipment", "Software updates"], "answer": "Malicious software"},
                {"question": "What is two-factor authentication (2FA)?", "options": ["Using two passwords", "Verification using two different methods", "Logging in from two devices", "Having two user accounts"], "answer": "Verification using two different methods"},
                {"question": "What is a vulnerability in cybersecurity?", "options": ["A weakness that can be exploited", "A type of virus", "A security tool", "A secure connection"], "answer": "A weakness that can be exploited"},
                {"question": "What is a DDoS attack?", "options": ["Data Destruction on Site", "Distributed Denial of Service", "Domain Disruption Service", "Data Download System"], "answer": "Distributed Denial of Service"},
                {"question": "What does VPN stand for?", "options": ["Very Private Network", "Virtual Private Network", "Virtual Personal Network", "Verified Processing Node"], "answer": "Virtual Private Network"},
                {"question": "What is a strong password typically characterized by?", "options": ["Using your name with numbers", "Short and simple to remember", "Long with a mix of character types", "Using common words"], "answer": "Long with a mix of character types"},
                {"question": "What is social engineering in cybersecurity?", "options": ["Building social networks securely", "Manipulating people to divulge information", "Engineering social media platforms", "Sociological analysis of security"], "answer": "Manipulating people to divulge information"},
                {"question": "What is the primary purpose of an antivirus program?", "options": ["Speed up your computer", "Detect and remove malicious software", "Improve internet connection", "Encrypt your files"], "answer": "Detect and remove malicious software"},
                {"question": "What is a data breach?", "options": ["Unauthorized access to data", "Data backup process", "Data transfer between systems", "Data compression technique"], "answer": "Unauthorized access to data"},
                {"question": "What is a cyber threat?", "options": ["A new cybersecurity tool", "A potential security violation", "A type of encryption", "A secure network"], "answer": "A potential security violation"},
                {"question": "What is a security patch?", "options": ["Physical security for servers", "Software update that addresses vulnerabilities", "A bandwidth limitation", "A type of firewall"], "answer": "Software update that addresses vulnerabilities"},
                {"question": "What is the purpose of a password manager?", "options": ["Managing network passwords", "Securely storing and generating passwords", "Resetting forgotten passwords", "Blocking unauthorized access"], "answer": "Securely storing and generating passwords"}
            ],
            "medium": [
                {"question": "What is a zero-day vulnerability?", "options": ["A vulnerability with no impact", "A vulnerability known for zero days", "A vulnerability unknown to the software vendor", "A vulnerability that takes zero days to exploit"], "answer": "A vulnerability unknown to the software vendor"},
                {"question": "What is the difference between symmetric and asymmetric encryption?", "options": ["Speed vs. security", "One key vs. two keys", "Software vs. hardware encryption", "Local vs. cloud encryption"], "answer": "One key vs. two keys"},
                {"question": "What is a man-in-the-middle attack?", "options": ["An attack from someone inside an organization", "An attack intercepting communications between two parties", "An attack on network infrastructure", "An attack from a middleman"], "answer": "An attack intercepting communications between two parties"},
                {"question": "What is a honeypot in cybersecurity?", "options": ["A sweet reward for finding vulnerabilities", "A trap set to detect unauthorized access", "A repository of stolen data", "A type of social engineering"], "answer": "A trap set to detect unauthorized access"},
                {"question": "What is the purpose of penetration testing?", "options": ["Testing physical security barriers", "Testing network connection speed", "Simulating cyberattacks to find vulnerabilities", "Testing user passwords"], "answer": "Simulating cyberattacks to find vulnerabilities"},
                {"question": "What is a botnet?", "options": ["A network of AI bots", "A network of compromised devices controlled remotely", "An automated network testing tool", "A network security protocol"], "answer": "A network of compromised devices controlled remotely"},
                {"question": "What is network segmentation?", "options": ["Dividing a network into segments for better management", "Breaking network connections", "Analyzing network traffic patterns", "Connecting multiple networks"], "answer": "Dividing a network into segments for better management"},
                {"question": "What is a brute force attack?", "options": ["A physical attack on servers", "Trying all possible passwords until finding the correct one", "Forcing a system shutdown", "Breaking encryption forcefully"], "answer": "Trying all possible passwords until finding the correct one"},
                {"question": "What is the purpose of a SIEM system?", "options": ["Security Information and Event Management", "Secure Internet Email Monitoring", "System Integration for Enterprise Management", "Server Infrastructure Evaluation Method"], "answer": "Security Information and Event Management"},
                {"question": "What is the principle of least privilege?", "options": ["Giving users minimal access rights needed for their work", "Using minimal security measures", "Restricting access to privileged users only", "Using low-cost security solutions"], "answer": "Giving users minimal access rights needed for their work"},
                {"question": "What is a security incident response plan?", "options": ["A plan for responding to security guard incidents", "A documented approach to addressing security breaches", "A layout of security camera placement", "A schedule for security updates"], "answer": "A documented approach to addressing security breaches"},
                {"question": "What is the purpose of encryption at rest?", "options": ["Encrypting inactive data in storage", "Encrypting data when computer is off", "Encrypting network traffic", "Encrypting backup files only"], "answer": "Encrypting inactive data in storage"},
                {"question": "What is a security audit?", "options": ["Checking physical security measures", "Systematic evaluation of security systems and policies", "Auditing user access times", "Checking for unauthorized software"], "answer": "Systematic evaluation of security systems and policies"},
                {"question": "What is the OWASP Top 10?", "options": ["Top 10 cybersecurity companies", "List of most common web application security risks", "Top 10 most secure applications", "10 steps to secure networks"], "answer": "List of most common web application security risks"},
                {"question": "What is a CSRF attack?", "options": ["Cross-Site Request Forgery", "Client Server Response Failure", "Critical Security Risk Factor", "Credentials Stored in Raw Format"], "answer": "Cross-Site Request Forgery"}
            ],
            "advanced": [
                {"question": "What is threat hunting?", "options": ["Searching for bounties on cybercriminals", "Proactively searching for threats in a network", "Monitoring external threat intelligence", "Tracking hacker forums"], "answer": "Proactively searching for threats in a network"},
                {"question": "What is a security operations center (SOC)?", "options": ["A physical security center", "A facility housing security teams and tools", "Software for security operations", "A security training center"], "answer": "A facility housing security teams and tools"},
                {"question": "What is the CIA triad in cybersecurity?", "options": ["Central Intelligence Agency triad", "Confidentiality, Integrity, and Availability", "Critical Infrastructure Assessment", "Computer Incident Analysis"], "answer": "Confidentiality, Integrity, and Availability"},
                {"question": "What is a rootkit?", "options": ["A gardening toolkit", "Software providing privileged access while hiding itself", "A toolkit for root cause analysis", "Software for managing root directories"], "answer": "Software providing privileged access while hiding itself"},
                {"question": "What is a secure enclave?", "options": ["A physical secure location", "A hardware-based isolated execution environment", "A secure network zone", "An encrypted data storage"], "answer": "A hardware-based isolated execution environment"},
                {"question": "What is devsecops?", "options": ["Developers securing operations", "Integrating security into DevOps processes", "Development security options", "Device security operations"], "answer": "Integrating security into DevOps processes"},
                {"question": "What is threat intelligence?", "options": ["Information about existing or emerging threats", "AI for threat detection", "Intelligent threat response", "Threats to intelligence agencies"], "answer": "Information about existing or emerging threats"},
                {"question": "What is a hardware security module (HSM)?", "options": ["Physical security for hardware", "A dedicated crypto-processing device", "A module for securing hardware connections", "Hardware monitoring software"], "answer": "A dedicated crypto-processing device"},
                {"question": "What is a security orchestration, automation and response (SOAR) platform?", "options": ["A platform for coordinating musicians", "A technology stack for security operations", "Software for automated security testing", "A response plan template"], "answer": "A technology stack for security operations"},
                {"question": "What is homomorphic encryption?", "options": ["Encryption that preserves shape", "Encryption allowing computation on encrypted data", "Encryption for home networks", "Encryption with similar keys"], "answer": "Encryption allowing computation on encrypted data"},
                {"question": "What is a supply chain attack?", "options": ["Attacking retail supply chains", "Compromising a target by attacking its supply network", "Disrupting supply deliveries", "Attacking chain stores"], "answer": "Compromising a target by attacking its supply network"},
                {"question": "What is the principle of defense in depth?", "options": ["Defending the deepest parts of a network", "Using multiple security controls in layers", "Deep analysis of defense strategies", "Defending against deep packet inspection"], "answer": "Using multiple security controls in layers"},
                {"question": "What is a security information and event management (SIEM) system used for?", "options": ["Managing security teams", "Collecting and analyzing security data from various sources", "Managing security events", "Managing security information disclosure"], "answer": "Collecting and analyzing security data from various sources"},
                {"question": "What is a sandbox in cybersecurity?", "options": ["A development environment", "An isolated environment for testing suspicious code", "A children's play area", "A backup storage location"], "answer": "An isolated environment for testing suspicious code"},
                {"question": "What is threat modeling?", "options": ["Creating physical models of threats", "A process for identifying potential threats", "Modeling the behavior of attackers", "A mathematical model of attack vectors"], "answer": "A process for identifying potential threats"}
            ]
        },
        "Cloud Computing": {
            "basic": [
                {"question": "What is cloud computing?", "options": ["Computing about weather", "Delivering computing services over the internet", "A type of weather forecasting", "Computing with fog machines"], "answer": "Delivering computing services over the internet"},
                {"question": "What is IaaS?", "options": ["Internet as a Service", "Infrastructure as a Service", "Installation as a Service", "Integration as a Service"], "answer": "Infrastructure as a Service"},
                {"question": "What is SaaS?", "options": ["Storage as a Service", "Software as a Service", "Security as a Service", "Systems as a Service"], "answer": "Software as a Service"},
                {"question": "What is PaaS?", "options": ["Programming as a Service", "Platform as a Service", "Process as a Service", "Product as a Service"], "answer": "Platform as a Service"},
                {"question": "What is a cloud deployment model?", "options": ["A way to deploy applications", "How cloud resources are made available to users", "A diagram of cloud infrastructure", "A method to deploy servers"], "answer": "How cloud resources are made available to users"},
                {"question": "What is a public cloud?", "options": ["A cloud available to the general public", "A government cloud", "A free cloud service", "A cloud for public records"], "answer": "A cloud available to the general public"},
                {"question": "What is a private cloud?", "options": ["A cloud for private individuals", "Cloud services used by a single organization", "A secure cloud environment", "A personal cloud storage"], "answer": "Cloud services used by a single organization"},
                {"question": "What is a hybrid cloud?", "options": ["A mix of different cloud providers", "A combination of public and private clouds", "A cloud with mixed services", "A partially implemented cloud"], "answer": "A combination of public and private clouds"},
                {"question": "What is virtualization in cloud computing?", "options": ["Creating virtual reality environments", "Creating virtual versions of IT resources", "Visualizing cloud resources", "Virtual meetings in the cloud"], "answer": "Creating virtual versions of IT resources"},
                {"question": "What is a virtual machine?", "options": ["A physical computer", "A software emulation of a computer", "A fast computer", "A computer used for virtual reality"], "answer": "A software emulation of a computer"},
                {"question": "What is scalability in cloud computing?", "options": ["Measuring cloud servers", "Ability to increase or decrease resources as needed", "Scaling pricing models", "Measuring performance"], "answer": "Ability to increase or decrease resources as needed"},
                {"question": "What is cloud storage?", "options": ["Storing data in actual clouds", "Storing data on remote servers accessed via internet", "A storage device shaped like a cloud", "Storing weather data"], "answer": "Storing data on remote servers accessed via internet"},
                {"question": "What is a cloud service provider?", "options": ["A weather service", "A company that offers cloud computing services", "A provider of internet services", "A provider of server hardware"], "answer": "A company that offers cloud computing services"},
                {"question": "What is multi-tenancy in cloud computing?", "options": ["Having multiple IT staff", "Multiple users sharing the same computing resources", "Having backup systems", "Renting multiple cloud services"], "answer": "Multiple users sharing the same computing resources"},
                {"question": "What is elasticity in cloud computing?", "options": ["Flexible pricing", "Ability to stretch virtual servers", "Automatic scaling of resources based on demand", "Physical flexibility of data centers"], "answer": "Automatic scaling of resources based on demand"}
            ],
            "medium": [
                {"question": "What is a container in cloud computing?", "options": ["A storage unit", "A standardized unit of software", "A shipping container for servers", "A data structure"], "answer": "A standardized unit of software"},
                {"question": "What is Docker?", "options": ["A cloud provider", "A containerization platform", "A networking tool", "A virtual machine manager"], "answer": "A containerization platform"},
                {"question": "What is Kubernetes?", "options": ["A cloud storage service", "A container orchestration system", "A programming language", "A cloud security tool"], "answer": "A container orchestration system"},
                {"question": "What is serverless computing?", "options": ["Computing without any servers", "A model where cloud provider manages the infrastructure", "Computing on client devices only", "A peer-to-peer computing model"], "answer": "A model where cloud provider manages the infrastructure"},
                {"question": "What is auto-scaling in cloud computing?", "options": ["Automatic pricing adjustments", "Automatically adjusting resources based on demand", "Self-scaling applications", "Automatic updates to software"], "answer": "Automatically adjusting resources based on demand"},
                {"question": "What is a load balancer in cloud computing?", "options": ["A device that balances server loads", "A pricing model", "A weight measurement tool", "A tool for balancing workloads"], "answer": "A device that balances server loads"},
                {"question": "What is cloud orchestration?", "options": ["Playing music in the cloud", "Automated arrangement and coordination of cloud resources", "Managing cloud security", "Scheduling cloud tasks"], "answer": "Automated arrangement and coordination of cloud resources"},
                {"question": "What is infrastructure as code?", "options": ["Programming on infrastructure", "Managing infrastructure through code rather than manual processes", "Code stored on infrastructure", "Infrastructure built with code components"], "answer": "Managing infrastructure through code rather than manual processes"},
                {"question": "What is a microservice architecture?", "options": ["Architecture for small services", "Breaking applications into small, independent services", "Microscopic service design", "Architecture for micro-computers"], "answer": "Breaking applications into small, independent services"},
                {"question": "What is cloud monitoring?", "options": ["Watching clouds in the sky", "Tracking cloud resource performance and availability", "Monitoring cloud service providers", "Watching for cloud security threats"], "answer": "Tracking cloud resource performance and availability"},
                {"question": "What is a service level agreement (SLA)?", "options": ["An agreement between service levels", "A contract defining expected service levels", "A service scheduling agreement", "A level of service architecture"], "answer": "A contract defining expected service levels"},
                {"question": "What is cloud migration?", "options": ["Moving clouds across the sky", "Moving applications and data to the cloud", "Migrating between cloud providers", "Seasonal movement of cloud resources"], "answer": "Moving applications and data to the cloud"},
                {"question": "What is a region in cloud computing?", "options": ["A geographical area", "A physical area containing data centers", "A logical division of cloud resources", "A market region for cloud services"], "answer": "A physical area containing data centers"},
                {"question": "What is a cloud-native application?", "options": ["An application born in the cloud", "An application designed specifically for cloud environments", "A natural cloud formation", "An application with cloud graphics"], "answer": "An application designed specifically for cloud environments"},
                {"question": "What is DevOps in the context of cloud computing?", "options": ["Development operations", "A set of practices combining development and operations", "Device operations", "Developer options"], "answer": "A set of practices combining development and operations"}
            ],
            "advanced": [
                {"question": "What is multi-cloud strategy?", "options": ["Using multiple cloud formations", "Using services from multiple cloud providers", "Having multiple clouds as backup", "A strategy for cloud multiplicity"], "answer": "Using services from multiple cloud providers"},
                {"question": "What is cloud governance?", "options": ["Government regulation of clouds", "Framework for managing cloud implementations", "Governing body for cloud standards", "Political control of cloud services"], "answer": "Framework for managing cloud implementations"},
                {"question": "What is a cloud-native network function (CNF)?", "options": ["A natural network function", "Network functions designed for cloud environments", "A cloud formation network", "Native networking in clouds"], "answer": "Network functions designed for cloud environments"},
                {"question": "What is a service mesh?", "options": ["A network of interlinked services", "A mesh networking protocol", "A service network topology", "A mesh of cloud providers"], "answer": "A network of interlinked services"},
                {"question": "What is immutable infrastructure?", "options": ["Infrastructure that cannot be changed", "Infrastructure that can't be destroyed", "Infrastructure replaced entirely when changes needed", "Permanent infrastructure"], "answer": "Infrastructure replaced entirely when changes needed"},
                {"question": "What is a cloud access security broker (CASB)?", "options": ["A security guard for cloud access", "A security policy enforcement point", "A broker for cloud services", "A security monitoring tool"], "answer": "A security policy enforcement point"},
                {"question": "What is FinOps?", "options": ["Financial operations", "Practice of managing cloud costs", "Financial optimization services", "Financial cloud applications"], "answer": "Practice of managing cloud costs"},
                {"question": "What is a cloud management platform?", "options": ["A viewing platform for clouds", "Software for managing cloud resources", "A management system in the cloud", "A platform for cloud service providers"], "answer": "Software for managing cloud resources"},
                {"question": "What is a cloud service broker?", "options": ["A person who sells cloud services", "An intermediary between cloud consumers and providers", "A broker who uses cloud services", "A negotiator for cloud contracts"], "answer": "An intermediary between cloud consumers and providers"},
                {"question": "What is cloud bursting?", "options": ["When a cloud overflows with data", "Deploying to public cloud when private cloud reaches capacity", "A security breach in cloud systems", "Rapid expansion of cloud resources"], "answer": "Deploying to public cloud when private cloud reaches capacity"},
                {"question": "What is a cloud-native security model?", "options": ["Security born in clouds", "Security designed specifically for cloud environments", "Natural security for clouds", "Native encryption in clouds"], "answer": "Security designed specifically for cloud environments"},
                {"question": "What is a cloud center of excellence (CCoE)?", "options": ["A prestigious cloud award", "A cross-functional team promoting cloud best practices", "A central cloud authority", "An excellence rating for clouds"], "answer": "A cross-functional team promoting cloud best practices"},
                {"question": "What is chaos engineering in cloud environments?", "options": ["Creating chaos in cloud systems", "Deliberately introducing failures to test resilience", "Engineering chaotic cloud formations", "Disorganized cloud architecture"], "answer": "Deliberately introducing failures to test resilience"},
                {"question": "What is a cloud operating model?", "options": ["A model airplane in clouds", "Framework defining how cloud resources are delivered and operated", "A model for operating in cloudy weather", "A model operating system in the cloud"], "answer": "Framework defining how cloud resources are delivered and operated"},
                {"question": "What is edge computing in relation to cloud computing?", "options": ["Computing at the edge of clouds", "Processing data closer to where it's generated", "Cutting-edge computing technology", "Computing at the boundaries of networks"], "answer": "Processing data closer to where it's generated"}
            ]
        },
        "Embedded Systems": {
            "basic": [
                {"question": "What is an embedded system?", "options": ["A computer system embedded in another device", "A system buried underground", "A system embedded in software", "A system within a system"], "answer": "A computer system embedded in another device"},
                {"question": "Which of these is an example of an embedded system?", "options": ["Desktop computer", "Cloud server", "Digital watch", "Mainframe computer"], "answer": "Digital watch"},
                {"question": "What is a microcontroller?", "options": ["A small controller device", "An integrated circuit containing a processor, memory, and I/O peripherals", "A microscopic control unit", "A control device for microscopes"], "answer": "An integrated circuit containing a processor, memory, and I/O peripherals"},
                {"question": "What language is commonly used for embedded systems programming?", "options": ["Python", "Java", "C", "Ruby"], "answer": "C"},
                {"question": "What is firmware?", "options": ["Hardware that's firmly attached", "Permanent software programmed into a read-only memory", "Firmly established protocols", "A firm operating system"], "answer": "Permanent software programmed into a read-only memory"},
                {"question": "What is real-time computing?", "options": ["Computing that happens instantly", "Computing with time constraints", "Computing in real-world scenarios", "Computing with real numbers"], "answer": "Computing with time constraints"},
                {"question": "What is an RTOS?", "options": ["Real-Time Operating System", "Remote Technology Operating System", "Runtime Optimization System", "Resource Tracking Operating System"], "answer": "Real-Time Operating System"},
                {"question": "What is a GPIO pin?", "options": ["Global Process Input/Output", "General Purpose Input/Output", "Graphics Processing Input/Output", "Generic Port Input/Output"], "answer": "General Purpose Input/Output"},
                {"question": "What is a sensor in embedded systems?", "options": ["A device that detects physical changes", "A control mechanism", "A display unit", "A processing unit"], "answer": "A device that detects physical changes"},
                {"question": "What is an actuator in embedded systems?", "options": ["A sensor type", "A component that controls a mechanism", "A type of processor", "A display mechanism"], "answer": "A component that controls a mechanism"},
                {"question": "What is the purpose of a bootloader?", "options": ["To load boots", "To initialize the hardware and load the operating system", "To load applications", "To restart the system"], "answer": "To initialize the hardware and load the operating system"},
                {"question": "What is flash memory?", "options": ["Memory that flashes", "Non-volatile memory that can be electrically erased and reprogrammed", "Very fast memory", "Memory with lights"], "answer": "Non-volatile memory that can be electrically erased and reprogrammed"},
                {"question": "What does IoT stand for?", "options": ["Internet of Technology", "Internet of Things", "Input Output Technology", "Integrated Online Technology"], "answer": "Internet of Things"},
                {"question": "What is an SoC?", "options": ["System of Computers", "System on Chip", "Service on Cloud", "Society of Computing"], "answer": "System on Chip"},
                {"question": "What is the purpose of an ADC in embedded systems?", "options": ["Advanced Digital Computing", "Analog-to-Digital Converter", "Automated Device Control", "Application Development Component"], "answer": "Analog-to-Digital Converter"}
            ],
            "medium": [
                {"question": "What is the difference between a microcontroller and a microprocessor?", "options": ["Size", "Microcontroller includes CPU, memory, and I/O in one chip", "Processing speed", "Programming language used"], "answer": "Microcontroller includes CPU, memory, and I/O in one chip"},
                {"question": "What is a watchdog timer?", "options": ["A timer that watches for dogs", "A mechanism to detect and recover from system malfunctions", "A timer for security systems", "A timer that monitors user activity"], "answer": "A mechanism to detect and recover from system malfunctions"},
                {"question": "What is interrupt handling in embedded systems?", "options": ["Stopping system execution", "Managing unexpected events during execution", "Interrupting the power supply", "Handling human interruptions"], "answer": "Managing unexpected events during execution"},
                {"question": "What is a hard real-time system?", "options": ["A system that's physically tough", "A system where timing deadlines must be met", "A difficult system to program", "A system with hard drives"], "answer": "A system where timing deadlines must be met"},
                {"question": "What is I2C in embedded systems?", "options": ["A type of processor", "A communication protocol", "A security feature", "A memory type"], "answer": "A communication protocol"},
                {"question": "What is a soft real-time system?", "options": ["A system made of soft materials", "A system where timing deadlines are desirable but not critical", "A system that's easy to program", "A system with flexible requirements"], "answer": "A system where timing deadlines are desirable but not critical"},
                {"question": "What is a DAC in embedded systems?", "options": ["Digital Audio Controller", "Digital-to-Analog Converter", "Direct Access Controller", "Data Acquisition Component"], "answer": "Digital-to-Analog Converter"},
                {"question": "What is SPI in embedded systems?", "options": ["System Performance Index", "Serial Peripheral Interface", "Synchronized Protocol Interface", "Standard Programming Interface"], "answer": "Serial Peripheral Interface"},
                {"question": "What is a task in an RTOS?", "options": ["A job for the system to do", "An assignment for the programmer", "A unit of execution", "A component of the operating system"], "answer": "A unit of execution"},
                {"question": "What is priority inversion in real-time systems?", "options": ["Reversing the priorities of tasks", "A scheduling anomaly where a higher priority task waits for a lower priority one", "Inverting the system priorities", "A programming technique"], "answer": "A scheduling anomaly where a higher priority task waits for a lower priority one"},
                {"question": "What is a PWM signal?", "options": ["Power Wave Management", "Pulse Width Modulation", "Process Work Module", "Programmable Wave Mechanism"], "answer": "Pulse Width Modulation"},
                {"question": "What is a memory-mapped I/O?", "options": ["A map of memory locations", "Treating I/O devices as memory locations", "Mapping memory to I/O devices", "A memory management technique"], "answer": "Treating I/O devices as memory locations"},
                {"question": "What is a CAN bus?", "options": ["Controller Area Network", "Computer Automated Network", "Central Access Node", "Control and Navigation"], "answer": "Controller Area Network"},
                {"question": "What is the purpose of a JTAG interface?", "options": ["Java Testing and Guidance", "Just Testing And Grading", "Joint Test Action Group interface for testing and debugging", "Java Tag Authentication Gateway"], "answer": "Joint Test Action Group interface for testing and debugging"},
                {"question": "What is a deadlock in embedded systems?", "options": ["A lock that cannot be opened", "A situation where tasks are waiting for resources held by each other", "A security feature", "A hardware failure"], "answer": "A situation where tasks are waiting for resources held by each other"}
            ],
            "advanced": [
                {"question": "What is the purpose of memory protection in embedded systems?", "options": ["Protecting memory from physical damage", "Preventing unauthorized memory access", "Extending memory life", "Adding more memory"], "answer": "Preventing unauthorized memory access"},
                {"question": "What is cache coherence?", "options": ["Organizing cache memory", "Consistency of data stored in local caches of shared resource", "Cache memory backup", "Speed of cache access"], "answer": "Consistency of data stored in local caches of shared resource"},
                {"question": "What is DMA in embedded systems?", "options": ["Direct Memory Access", "Digital Memory Array", "Data Management Application", "Dynamic Memory Allocation"], "answer": "Direct Memory Access"},
                {"question": "What is the difference between von Neumann and Harvard architectures?", "options": ["Different inventors", "Harvard has separate data and instruction memory", "Von Neumann is newer", "Harvard is only used in universities"], "answer": "Harvard has separate data and instruction memory"},
                {"question": "What is a semaphore in embedded systems?", "options": ["A type of signal", "A variable used to control access to shared resources", "A safety protocol", "A hardware component"], "answer": "A variable used to control access to shared resources"},
                {"question": "What is stack overflow in embedded systems?", "options": ["Too many questions on Stack Overflow website", "When a stack exceeds its allocated memory", "When the system overflows with tasks", "When the processor overheats"], "answer": "When a stack exceeds its allocated memory"},
                {"question": "What is an FPGA?", "options": ["Fast Processing Graphics Array", "Field-Programmable Gate Array", "Forward Power Gain Amplifier", "Flexible Processor Gateway Architecture"], "answer": "Field-Programmable Gate Array"},
                {"question": "What is a mutex in embedded systems?", "options": ["Multiple Execution", "Mutual Exclusion object for synchronization", "Mutable Extension", "Memory Utility Extension"], "answer": "Mutual Exclusion object for synchronization"},
                {"question": "What is boundary scan testing?", "options": ["Testing system boundaries", "A method for testing interconnects on PCBs", "Scanning for boundary errors", "Testing memory boundaries"], "answer": "A method for testing interconnects on PCBs"},
                {"question": "What is rate-monotonic scheduling?", "options": ["Scheduling based on data rates", "A static-priority scheduling algorithm", "Scheduling based on task monotony", "Scheduling based on processor rates"], "answer": "A static-priority scheduling algorithm"},
                {"question": "What is a cyclic executive in embedded systems?", "options": ["A rotating manager", "A simple scheduling approach using a predetermined fixed schedule", "An executive who works in cycles", "A cycling power management system"], "answer": "A simple scheduling approach using a predetermined fixed schedule"},
                {"question": "What is the purpose of an ASIC?", "options": ["Application-Specific Integrated Circuit", "Advanced System Interface Controller", "Analog Signal Input Converter", "Automated System Integration Component"], "answer": "Application-Specific Integrated Circuit"},
                {"question": "What is the difference between polling and interrupts?", "options": ["Polling is for elections, interrupts are for stopping", "Polling continuously checks for events, interrupts signal when events occur", "Polling is faster, interrupts are slower", "Polling uses more memory, interrupts use less"], "answer": "Polling continuously checks for events, interrupts signal when events occur"},
                {"question": "What is a race condition in embedded systems?", "options": ["A competition between systems", "An error when a system's behavior depends on sequence or timing of events", "A condition for racing games", "A high-speed processing mode"], "answer": "An error when a system's behavior depends on sequence or timing of events"},
                {"question": "What is the purpose of a BSP (Board Support Package)?", "options": ["Supporting circuit boards physically", "Providing software support for a specific hardware board", "Business Support Protocol", "Basic System Provider"], "answer": "Providing software support for a specific hardware board"}
            ]
        },
        "Blockchain": {
            "basic": [
                {"question": "What is blockchain?", "options": ["A type of database", "A cryptocurrency", "A programming language", "A type of network"], "answer": "A type of database"},
                {"question": "What is a smart contract?", "options": ["A legal document", "A self-executing contract with terms written in code", "A type of cryptocurrency", "A blockchain protocol"], "answer": "A self-executing contract with terms written in code"}
            ],
            "medium": [
                {"question": "What is proof of work?", "options": ["A consensus mechanism", "A type of blockchain", "A programming algorithm", "A cryptographic key"], "answer": "A consensus mechanism"}
            ],
            "advanced": [
                {"question": "What is sharding in blockchain?", "options": ["A method to split data", "A consensus mechanism", "A type of cryptographic key", "A way to scale blockchains"], "answer": "A way to scale blockchains"}
            ]
        },
        "Game Development": {
            "basic": [
                {"question": "What is a game engine?", "options": ["A hardware device", "A software framework for game development", "A type of programming language", "A graphics card"], "answer": "A software framework for game development"}
            ],
            "medium": [
                {"question": "What is a sprite in game development?", "options": ["A character in a game", "A 2D image or animation", "A type of game engine", "A programming tool"], "answer": "A 2D image or animation"}
            ],
            "advanced": [
                {"question": "What is ray tracing?", "options": ["A rendering technique for realistic lighting", "A type of game engine", "A programming language", "A debugging tool"], "answer": "A rendering technique for realistic lighting"}
            ]
        },
        "Robotics": {
            "basic": [
                {"question": "What is a robot?", "options": ["A machine capable of carrying out tasks", "A type of computer", "A programming language", "A type of AI"], "answer": "A machine capable of carrying out tasks"}
            ],
            "medium": [
                {"question": "What is inverse kinematics?", "options": ["A method to calculate joint angles for a desired position", "A type of robot", "A programming algorithm", "A hardware component"], "answer": "A method to calculate joint angles for a desired position"}
            ],
            "advanced": [
                {"question": "What is SLAM in robotics?", "options": ["Simultaneous Localization and Mapping", "A type of robot", "A programming language", "A hardware component"], "answer": "Simultaneous Localization and Mapping"}
            ]
        },
        "IoT (Internet of Things)": {
            "basic": [
                {"question": "What is IoT?", "options": ["A network of interconnected devices", "A type of programming language", "A type of database", "A type of AI"], "answer": "A network of interconnected devices"}
            ],
            "medium": [
                {"question": "What is MQTT in IoT?", "options": ["A lightweight messaging protocol", "A type of IoT device", "A programming language", "A hardware component"], "answer": "A lightweight messaging protocol"}
            ],
            "advanced": [
                {"question": "What is edge computing in IoT?", "options": ["Processing data closer to the source", "A type of IoT device", "A programming language", "A hardware component"], "answer": "Processing data closer to the source"}
            ]
        }
    }

    # Display quiz questions
    if selected_domain in quiz_questions:
        st.markdown(f"<h2 style='color: #4CAF50;'>Quiz: {selected_domain}</h2>", unsafe_allow_html=True)
        score = 0
        total_questions = 0
        user_answers = []

        # Iterate through difficulty levels
        for level, questions in quiz_questions[selected_domain].items():
            st.markdown(f"<h3 style='color: #2196F3;'>{level.capitalize()} Level</h3>", unsafe_allow_html=True)
            for i, question in enumerate(questions[:25]):  # Limit to 25 questions per subsection
                st.markdown(
                    f"""
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                        <p style="color: black;"><strong>Q{i+1} ({level.capitalize()}):</strong> {question['question']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                user_answer = st.radio(
                    f"Options for Q{i+1}:",
                    question["options"],
                    key=f"{level}_q{i+1}",
                    label_visibility="collapsed"
                )
                user_answers.append((user_answer, question["answer"]))
                total_questions += 1

        # Submit button for the entire quiz
        if st.button("Submit Quiz"):
            # Calculate score
            for user_answer, correct_answer in user_answers:
                if user_answer == correct_answer:
                    score += 1

            # Display final score
            st.markdown(
                f"""
                <div style="border: 2px solid #4CAF50; padding: 20px; margin-top: 20px; border-radius: 10px; background-color: ; color: black">
                    <h2 style="color: #4CAF50;">Your Final Score: {score}/{total_questions}</h2>
                </div>
                """, unsafe_allow_html=True
            )

            # Generate career roadmap using Gemini API
            with st.spinner("Fetching recommendations from Gemini..."):
                roadmap_prompt = f"Generate a detailed career roadmap for an engineering student from a Telangana college who is interested in {selected_domain} to become successful in his career, start from beginner to advanced level. The student scored {score} out of {total_questions} in a domain-specific quiz conducted by our app. Suggest best course of action for him to become successful in 2025. Be specific and include the following points: 1. Skills to learn 2. Certifications to pursue 3. Projects to work on 4. Internships or job opportunities to consider 5. Networking opportunities 6. Recommended resources (books, websites, etc.) 7. Any other relevant advice."
            response_text = text_model.generate_content(roadmap_prompt)
            st.subheader("Career Roadmap:")
            st.write(response_text.text)