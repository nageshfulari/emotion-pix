🎭 Emotionix – Emotion-Based Movie Recommendation System

Emotionix is an AI-powered web application developed by me that detects human facial emotions from images and recommends movies based on the detected emotional state. This project demonstrates my skills in Python, Flask, computer vision, machine learning integration, and full-stack web development.

🚀 Project Overview

I designed and built Emotionix to explore the practical application of emotion recognition using computer vision and to integrate it with a real-world recommendation system. The application captures facial images, analyzes emotions using a deep learning–based model, and maps the detected emotion to relevant movie genres using external APIs.

✨ Key Features

🔐 Implemented secure user authentication using Supabase

😊 Integrated facial emotion detection using FER (Facial Emotion Recognition)

🎬 Developed emotion-based movie recommendation logic

🔍 Implemented movie search functionality using IMDb API via RapidAPI

🎙️ Added voice-based movie search for enhanced usability

📷 Implemented camera on/off toggle for user privacy

🗄️ Designed an efficient SQLite caching system to reduce API calls and improve performance

🛠️ Technology Stack

Programming Language

Python

Backend

Flask

Frontend

HTML

CSS

JavaScript

Computer Vision & AI

OpenCV

FER (Facial Emotion Recognition)

APIs & Services

Supabase (Authentication)

RapidAPI – IMDb236 (Movie Data)

Database

SQLite (Caching)

⚙️ Installation & Local Setup
Prerequisites

Python 3.11 or higher

pip

Clone the Repository
git clone <your-repo-url>
cd Emotionix

Create & Activate Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate

Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

Environment Configuration

Create a .env file in the project root:

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
FLASK_SECRET_KEY=your_flask_secret_key
RAPIDAPI_KEY=your_rapidapi_key
RAPIDAPI_HOST=imdb236.p.rapidapi.com

Run the Application
python app.py


Access the application at:

http://localhost:5000

📁 Project Structure
Emotionix/
├── app.py                # Main Flask application
├── config.py             # Environment configuration handling
├── requirements.txt      # Project dependencies
├── templates/            # HTML templates
│   ├── home.html
│   ├── login.html
│   └── register.html
├── static/               # CSS, JavaScript, images
├── movie_cache.db        # SQLite cache database
└── .env                  # Environment variables (ignored)

🎯 Emotion-to-Genre Mapping Logic
Emotion	Genre
Happy	Comedy
Sad	Drama
Angry	Action
Surprise	Adventure
Neutral	Drama
Fear	Horror
📈 Learning Outcomes

Through this project, I gained hands-on experience in:

Building full-stack web applications using Flask

Integrating machine learning models into production-ready APIs

Working with image processing and facial emotion recognition

API integration and response caching

Secure authentication and environment variable management

🚀 Deployment

The application is structured to be deployment-ready on cloud platforms such as Render. Environment variables are securely managed, and optional services degrade gracefully if unavailable.

📜 License

This project is licensed under the MIT License.

👤 Author

Developed by: Nagesh Fulari | Ganesh Mane | Yashwardhan Mahamuni
Computer Vision | Machine Learning | Python | Flask
