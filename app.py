"""Emotionix Flask application.

Provides authentication, emotion detection from uploaded images,
and emotion-based movie recommendations with SQLite caching.
"""

# ============================================================================
# IMPORTS - Web Framework, Security, APIs, ML, Database
# ============================================================================

# Web Framework & HTTP utilities - Flask for web app structuring
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session

# Data processing & utilities - string matching, JSON handling, file paths, time
import re
import json
import os
from datetime import datetime, timedelta
import time

# Session management & configuration - store user sessions, load environment config
from flask_session import Session
import config
from dotenv import load_dotenv

# Authentication & Security - Supabase for user auth, password hashing, email validation
from supabase import create_client
from werkzeug.security import generate_password_hash, check_password_hash
from gotrue.errors import AuthApiError
from email_validator import validate_email, EmailNotValidError

# HTTP requests & External APIs - fetch movie data from RapidAPI (IMDb236)
import requests

# Computer Vision & Machine Learning - OpenCV for image processing, NumPy for arrays, FER for emotion detection
import cv2
import numpy as np

# Database - SQLite for caching movies and search results locally
import sqlite3

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================
# Load .env file early so config.py and app setup can use environment variables

load_dotenv()
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

USER_DATA_FILE = 'users.json'
supabase_client = None
supabase_init_error = None

# Initialize Supabase once at startup; keep app running even if auth config is missing.
try:
    # Prefer config.py values, but allow .env/runtime overrides for container/cloud deployments.
    supabase_url = getattr(config, 'SUPABASE_URL', None) or os.getenv('SUPABASE_URL')
    supabase_key = getattr(config, 'SUPABASE_KEY', None) or os.getenv('SUPABASE_KEY')
    if supabase_url and supabase_key:
        supabase_url = str(supabase_url).strip()
        supabase_key = str(supabase_key).strip()

        if supabase_url and supabase_key and len(supabase_url) > 0 and len(supabase_key) > 0:
            if supabase_key.startswith('eyJ') and len(supabase_key) > 100:
                if len(supabase_key) > 500:
                    print(
                        "Warning: SUPABASE_KEY appears to be a service_role key. Use the 'anon public' key instead for client-side authentication.")
            elif not supabase_key.startswith('eyJ'):
                print(f"Warning: SUPABASE_KEY format doesn't match expected JWT format.")
                print(f"  Current key starts with: '{supabase_key[:20]}...' (length: {len(supabase_key)})")
                print(f"  Expected: JWT token starting with 'eyJ' (200-300 characters)")
                print(f"  Action: Go to Supabase Dashboard → Settings → API → Copy 'anon public' key")
                supabase_init_error = f"SUPABASE_KEY format invalid. Current key starts with '{supabase_key[:20]}...' but should be a JWT token starting with 'eyJ'. Get the 'anon public' key from Supabase Dashboard → Settings → API."

            try:
                if not supabase_url or not supabase_key:
                    raise ValueError("SUPABASE_URL and SUPABASE_KEY must both be provided")
                supabase_client = create_client(supabase_url, supabase_key)

                print("Supabase client initialized successfully")
                print(f"Supabase URL: {supabase_url[:30]}...")
            except (ValueError, TypeError, Exception) as client_error:
                error_msg = str(client_error)
                print(f"Warning: Could not create Supabase client: {error_msg}")
                # Provide more helpful error messages for common issues
                if "Invalid API key" in error_msg or "invalid" in error_msg.lower() and "key" in error_msg.lower():
                    supabase_init_error = "Invalid Supabase API key. Please verify your SUPABASE_KEY environment variable in Render dashboard matches your Supabase project's anon/public key."
                elif "url" in error_msg.lower() or "connection" in error_msg.lower():
                    supabase_init_error = f"Supabase connection error: {error_msg}. Please verify your SUPABASE_URL is correct."
                elif "proxy" in error_msg.lower() or "unexpected keyword" in error_msg.lower():
                    supabase_init_error = f"Supabase client initialization error: {error_msg}. Ensure requirements.txt has httpx==0.24.1 and httpcore==0.17.3. This is a known compatibility issue with supabase-py 2.4.0."
                else:
                    supabase_init_error = error_msg
                supabase_client = None
        else:
            error_msg = "SUPABASE_URL or SUPABASE_KEY is empty"
            print(f"Warning: {error_msg}. Authentication features will be disabled.")
            supabase_init_error = error_msg
    else:
        error_msg = "SUPABASE_URL or SUPABASE_KEY not set in environment variables"
        print(f"Warning: {error_msg}. Authentication features will be disabled.")
        supabase_init_error = error_msg
except Exception as e:
    error_msg = f"Error during Supabase initialization: {str(e)}"
    print(f"Warning: {error_msg}")
    print("Authentication features will be disabled.")
    supabase_init_error = error_msg
    supabase_client = None

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


# ============================================================================
# DATABASE LAYER - Cache & User Storage
# ============================================================================
# Functions for managing SQLite cache (movies, search results) and user data

def load_users():
    """Load local fallback users from JSON storage."""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}


DATABASE = 'movie_cache.db'


def init_db():
    """Create cache tables if they do not exist."""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            # Separate caches keep frequent emotion lookups and text search results independent.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movie_cache (
                    genre TEXT PRIMARY KEY,
                    movies TEXT,
                    timestamp TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    search_query TEXT PRIMARY KEY,
                    results TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")


def get_cached_movies(genre):
    """Return cached movies for a genre if cache is still fresh."""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT movies, timestamp FROM movie_cache WHERE genre = ?', (genre,))
            row = cursor.fetchone()

            if row:
                movies_json = row[0]
                timestamp = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
                # Keep cache short-lived so recommendations stay fresh without hammering the API.
                if datetime.now() - timestamp < timedelta(hours=1):
                    return json.loads(movies_json)
    except Exception as e:
        print(f"Error getting cached movies: {e}")
    return None


def store_cached_movies(genre, movies):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                REPLACE INTO movie_cache (genre, movies, timestamp) 
                VALUES (?, ?, ?)
            ''', (genre, json.dumps(movies), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
    except Exception as e:
        print(f"Error storing cached movies: {e}")


def save_users(users):
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(users, f)
    except Exception as e:
        print(f"Error saving users: {e}")


# ============================================================================
# VALIDATION HELPERS - Email & Password Checks
# ============================================================================
# Functions to validate user input (email format, password strength)

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False


def is_valid_password(password: str) -> bool:
    """
    Validate password with regex:
    - Minimum 8 characters
    - At least one lowercase letter
    - At least one uppercase letter
    - At least one digit
    - At least one special character
    """
    if not password:
        return False
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^\w\s]).{8,}$'
    return re.match(pattern, password) is not None


def password_issues(password: str) -> list:
    """Return a list of human-readable issues for the given password."""
    issues = []
    if not password or len(password) < 8:
        issues.append('At least 8 characters long')
    if not re.search(r'[a-z]', password):
        issues.append('Include at least one lowercase letter')
    if not re.search(r'[A-Z]', password):
        issues.append('Include at least one uppercase letter')
    if not re.search(r'\d', password):
        issues.append('Include at least one digit')
    if not re.search(r'[^\w\s]', password):
        issues.append('Include at least one special character (e.g. !@#$%)')
    return issues


def create_user(email, password):
    users = load_users()
    if email in users:
        return False
    hashed_password = generate_password_hash(password)
    users[email] = hashed_password
    save_users(users)
    return True


def validate_user(email, password):
    users = load_users()
    if email in users and check_password_hash(users[email], password):
        return True
    return False


# ============================================================================
# AUTHENTICATION ROUTES - User Registration, Login, Logout
# ============================================================================
# Routes for user authentication: register, login, logout, email confirmation

@app.route('/resend_confirmation', methods=['POST'])
def resend_confirmation():
    """Resend email confirmation link if user hasn't verified yet (Supabase auth)."""
    if not supabase_client:
        error_msg = "Authentication service is not available."
        if supabase_init_error:
            error_msg += f" {supabase_init_error}"
        else:
            error_msg += " Please check your SUPABASE_URL and SUPABASE_KEY environment variables in your Render dashboard."
        flash(error_msg, "danger")
        return redirect(url_for('login'))

    email = request.form.get('email', '').strip()
    if not email:
        flash("Email is required.", "danger")
        return redirect(url_for('login'))
    try:
        response = supabase_client.auth.api.resend_confirmation(email)
        if "error" in response:
            flash("Error sending confirmation email: " + response["error"]["message"], "danger")
        else:
            flash("A new confirmation email has been sent. Please check your inbox.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page and form handler. Validates email and password strength."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not is_valid_email(email):
            flash('Invalid email format. Please enter a valid email.', 'danger')
            return render_template('register.html', email=email)

        issues = password_issues(password)
        if issues:
            suggestion_html = '<p>Password does not meet the following requirements:</p><ul>' + ''.join(
                f'<li>{issue}</li>' for issue in issues) + '</ul>'
            flash(suggestion_html, 'danger')
            return render_template('register.html', email=email)

        response = safe_supabase_sign_up(email, password)

        if response and "error" in response:
            flash("Error: " + response["error"]["message"], "danger")
        else:
            flash('Registration successful! Check your email to verify.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


def safe_supabase_sign_up(email, password):
    """Sign up with retries to handle transient Supabase/network failures."""
    if not supabase_client:
        error_msg = "Authentication service is not available."
        if supabase_init_error:
            error_msg += f" {supabase_init_error}"
        else:
            error_msg += " Please check your SUPABASE_URL and SUPABASE_KEY environment variables in your Render dashboard."
        return {"error": {"message": error_msg}}

    retries = 3
    for attempt in range(retries):
        try:
            return supabase_client.auth.sign_up({"email": email, "password": password})
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"Supabase Error after {retries} attempts: {str(e)}")
                return {"error": {"message": str(e)}}
    print('Failed to connect to the server after retries.')
    return {"error": {"message": "Failed to connect to the server. Please try again later."}}


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and form handler. Authenticates user with Supabase or local JSON."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if not email or not password:
            flash("Email and password are required!", "danger")
            return render_template('login.html')
        if not supabase_client:
            error_msg = "Authentication service is not available."
            if supabase_init_error:
                error_msg += f" {supabase_init_error}"
            else:
                error_msg += " Please check your SUPABASE_URL and SUPABASE_KEY environment variables in your Render dashboard."
            flash(error_msg, "danger")
            return render_template('login.html')

        try:
            response = supabase_client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if hasattr(response, 'error') and response.error:
                error_message = response.error.get("message", "Unknown error")
                if "Email not confirmed" in error_message:
                    flash("Your email is not confirmed. Please check your inbox.", "warning")
                    return redirect(url_for('login'))
                flash(f"Login failed: {error_message}", "danger")
            else:
                user = response.user
                session['user'] = {
                    'id': user.id,
                    'email': user.email,
                    'user_metadata': user.user_metadata,
                }
                flash('Login successful!', 'success')
                return redirect(url_for('home'))

        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clear user session and redirect to login."""
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.after_request
def add_no_cache_headers(response):
    """Prevent browser caching of authenticated pages to protect user privacy."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# ============================================================================
# EMOTION RECOGNITION & MOVIE RECOMMENDATIONS SETUP
# ============================================================================
# Configuration for FER (Facial Emotion Recognition) and RapidAPI movie data

emotion_to_genre = {
    "happy": "Comedy",
    "sad": "Drama",
    "anger": "Action",
    "fear": "Horror",
    "surprise": "Adventure",
    "neutral": "Drama"
}

# Load RapidAPI credentials for IMDb movie search
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST')

if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
    print("Warning: RAPIDAPI_KEY or RAPIDAPI_HOST not set. Movie recommendations may not work.")

# Initialize FER model at startup. If it fails, emotion detection will return 'neutral'.
emotion_detector = None

try:
    from fer import FER

    emotion_detector = FER(mtcnn=False)
    print("✅ FER initialized successfully")
except Exception as e:
    print("❌ FER initialization failed:", e)
    emotion_detector = None

# Map FER emotion labels to a normalized set
EMOTION_MAP = {
    "angry": "anger",
    "disgust": "anger",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral"
}


# ============================================================================
# EMOTION DETECTION FUNCTIONS
# ============================================================================
# Core logic: load image bytes, detect face(s), return strongest emotion

def detect_emotion(image_data):
    """Decode image bytes and return normalized emotion label."""
    if emotion_detector is None:
        print("⚠️ FER not initialized")
        return "neutral"

    try:
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️ Frame decode failed")
            return "neutral"
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))

        results = emotion_detector.detect_emotions(frame)

        if not results:
            print("⚠️ No face detected")
            return "neutral"
        best_emotion = "neutral"
        best_score = 0.0

        # Pick the highest confidence emotion across all detected faces.
        for face in results:
            for emotion, score in face["emotions"].items():
                if score > best_score:
                    best_score = score
                    best_emotion = emotion

        print("🎯 Emotion scores:", results)
        print("🎯 Selected emotion:", best_emotion, best_score)

        return EMOTION_MAP.get(best_emotion, "neutral")

    except Exception as e:
        print("❌ FER ERROR:", e)
        return "neutral"


# ============================================================================
# MOVIE RECOMMENDATION FUNCTIONS
# ============================================================================
# Fetch movies by genre from RapidAPI IMDb, with SQLite caching

def get_movie_recommendations(genre):
    """Fetch movies by genre from cache first, then RapidAPI on cache miss."""
    cached_movies = get_cached_movies(genre)
    if cached_movies:
        return cached_movies

    if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
        print("RAPIDAPI_KEY or RAPIDAPI_HOST not configured")
        return []

    url = "https://imdb236.p.rapidapi.com/api/imdb/search"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    params = {
        "type": "movie",
        "genre": genre,
        # Apply quality and language filters to return stronger Hindi movie suggestions.
        "rows": 100,
        "sortField": "startYear",
        "sortOrder": "DESC",
        "countriesOfOrigin": ["IN"],
        "spokenLanguages": ["hi"],
        "averageRatingFrom": "7",
        "averageRatingTo": "10",
        "numVotesFrom": "1000",
        "numVotesTo": "1000000",
        "startYearFrom": "1970",
        "startYearTo": "2026",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        print("RAPIDAPI STATUS:", response.status_code)

        response.raise_for_status()
        movies = response.json().get('results', [])
        store_cached_movies(genre, movies.copy())
        return movies
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movies: {e}")
        return []



# ============================================================================
# API ROUTES - Image Upload & Movie Data
# ============================================================================
# REST endpoints for emotion detection and movie recommendations

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_from_image():
    """Accept uploaded image and return detected emotion."""
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'success': False, 'message': 'Invalid image format. Only PNG, JPG, or JPEG allowed.'}), 400

    image_data = image_file.read()
    emotion = detect_emotion(image_data)
    if emotion:
        return jsonify({'success': True, 'emotion': emotion})
    else:
        return jsonify({'success': False, 'message': 'Emotion detection failed'})


@app.route('/get_movies', methods=['GET'])
def get_movies():
    """Get movie recommendations based on detected emotion query parameter."""
    emotion = request.args.get('emotion', 'happy')
    genre = emotion_to_genre.get(emotion, 'Comedy')
    response = get_movie_recommendations(genre)
    if not response:
        return jsonify({'movies': []})
    movies = []

    for movie in response:
        if not isinstance(movie, dict):
            continue
        primary_title = movie.get('primaryTitle')
        if not primary_title:
            continue
        description = movie.get('description', 'No description available.')
        primary_image = movie.get('primaryImage', None)
        trailer_url = movie.get('trailerUrl', None)
        if description and len(description) > 100:
            description = description[:97] + '...'
        if primary_image and description:
            movies.append({
                'primaryTitle': primary_title,
                'description': description,
                'primaryImage': primary_image,
                "trailerUrl": f"https://www.youtube.com/results?search_query={primary_title.replace(' ', '+')}+trailer"
            })
    return jsonify({'movies': movies})


def get_cached_search_results(search_query):
    """Return cached movie search results if within TTL."""
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT results, timestamp FROM search_cache WHERE search_query = ?', (search_query,))
            row = cursor.fetchone()

            if row:
                results_json = row[0]
                timestamp = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
                if datetime.now() - timestamp < timedelta(hours=1):
                    return json.loads(results_json)
    except Exception as e:
        print(f"Error getting cached search results: {e}")
    return None


def store_cached_search_results(search_query, results):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                REPLACE INTO search_cache (search_query, results, timestamp) 
                VALUES (?, ?, ?)
            ''', (search_query, json.dumps(results), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
    except Exception as e:
        print(f"Error storing cached search results: {e}")


@app.route('/search_movie', methods=['GET'])
def search_movie():
    """Search movies by title (query param) with response caching."""
    search_query = request.args.get('query', '').strip()
    if not search_query:
        return jsonify({'movies': []})

    # Cache by raw query to avoid repeated calls while user types similar searches.
    cached_results = get_cached_search_results(search_query)
    if cached_results:
        return jsonify({'movies': cached_results})

    if not RAPIDAPI_KEY or not RAPIDAPI_HOST:
        print("RAPIDAPI_KEY or RAPIDAPI_HOST not configured")
        return jsonify({'movies': []})

    url = "https://imdb236.p.rapidapi.com/api/imdb/search"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    params = {
        "primaryTitleAutocomplete": search_query,
        "type": "movie",
        "rows": "25",
        "sortOrder": "ASC",
        "sortField": "id",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        print("RAPIDAPI STATUS:", response.status_code)

        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error searching movies: {e}")
        return jsonify({'movies': []})

    movies = []
    if 'results' in data:
        for movie in data['results']:
            if not isinstance(movie, dict):
                continue
            primary_title = movie.get('primaryTitle')
            if not primary_title:
                continue
            description = movie.get('description', 'No description available.')
            primary_image = movie.get('primaryImage', None)
            trailer_url = movie.get('trailerUrl', None)

            if description and len(description) > 100:
                description = description[:97] + '...'

            if primary_image and description:
                movies.append({
                    'primaryTitle': primary_title,
                    'description': description,
                    'primaryImage': primary_image,
                    'trailerUrl': trailer_url or f"https://www.youtube.com/results?search_query={primary_title.replace(' ', '+')}+trailer"
                })

    store_cached_search_results(search_query, movies)

    return jsonify({'movies': movies})


# ============================================================================
# PAGE ROUTES - Home, Login, Health Check
# ============================================================================
# Routes for rendering HTML pages and diagnostic endpoints

@app.route('/')
def home_check():
    """Redirect to home if logged in, otherwise to login page."""
    if 'user' in session:
        return redirect("/home")
    else:
        return redirect('/login')


@app.route('/home')
def home():
    """Main page showing emotion detection and movie recommendations."""
    if 'user' not in session:
        flash("You need to log in first.", "warning")
        return redirect(url_for('login'))
    return render_template('home.html', user=session['user'])


@app.route('/health')
def health_check():
    """Diagnostic endpoint: reports Supabase and RapidAPI configuration status."""
    status = {
        'supabase_configured': supabase_client is not None,
        'supabase_url_set': bool(getattr(config, 'SUPABASE_URL', None) or os.getenv('SUPABASE_URL')),
        'supabase_key_set': bool(getattr(config, 'SUPABASE_KEY', None) or os.getenv('SUPABASE_KEY')),
        'init_error': supabase_init_error if supabase_init_error else None
    }
    return jsonify(status)


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)