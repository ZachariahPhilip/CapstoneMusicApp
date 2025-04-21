from flask import Flask, request, redirect, session, jsonify, url_for
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pymongo import MongoClient
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'secretkey'  # 🔒 Replace this with something random and secure!

# ✅ Session Configurations (IMPORTANT for localhost)
app.config['SESSION_COOKIE_NAME'] = 'spotify-login-session'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_SAMESITE'] = "None"

# Spotify API Credentials (PUT YOUR REAL KEYS HERE)
SPOTIPY_CLIENT_ID = 'adae3d81cb654f98abe6eeeebdba6fac'
SPOTIPY_CLIENT_SECRET = '0471938d2f66466fb9f172b398eadf0b'
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'

# Setup Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope='user-library-read user-read-recently-played'
)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["music_recommendation"]
collection = db["spotify_tracks"]

# Flask Routes
@app.route('/')
def index():
    return "🎵 Welcome to the Music Recommender! Visit /login to get started."

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('success'))

@app.route('/success')
def success():
    return "<h1>✅ Login Successful! You can now pull your Spotify tracks.</h1>"

@app.route('/logout')
def logout():
    session.clear()
    return "<h1>👋 You have been logged out.</h1>"

@app.route('/get_recent_tracks')
def get_recent_tracks():
    token_info = session.get('token_info', None)
    if not token_info:
        return redirect('/login')

    sp = spotipy.Spotify(auth=token_info['access_token'])
    results = sp.current_user_recently_played(limit=10)

    print("SPOTIFY RESULTS:")
    print(results)  # 🔥 Debug printout in terminal

    # Save each track to MongoDB
    for item in results.get('items', []):
        track = item['track']
        track_info = {
            "track_name": track['name'],
            "artist": track['artists'][0]['name'],
            "album": track['album']['name'],
            "popularity": track['popularity'],
            "spotify_id": track['id']
        }
        collection.insert_one(track_info)

    return "<h1>🎵 Recent tracks successfully saved to MongoDB!</h1>"

if __name__ == '__main__':
    app.run(debug=True)
