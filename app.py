from flask import Flask, jsonify
from pymongo import MongoClient
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

app = Flask(__name__)

#  MongoDB Connection
import os
from pymongo import MongoClient

mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(mongodb_uri)
db = client["music_recommendation"]
collection = db["spotify_tracks"]


#  Load data into a Pandas DataFrame from MongoDB
df = pd.DataFrame(list(collection.find()))
if not df.empty:
    df.drop('_id', axis=1, inplace=True)  # Drop MongoDB Object ID field

print(" Data Loaded Successfully from MongoDB:")
print(df.head())  # Prints first 5 rows

#  Encode song IDs and user IDs
song_encoder = LabelEncoder()
df['song_id'] = song_encoder.fit_transform(df['track_name'])  # Encode song names

# Simulating users with artist names (since we don't have real user data)
user_encoder = LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['artist'])

#  Convert to PyTorch tensors
user_tensor = torch.LongTensor(df['user_id'].values)
song_tensor = torch.LongTensor(df['song_id'].values)
popularity_tensor = torch.FloatTensor(df['popularity'].values) / 100.0  # Normalize

#  Print to confirm encoding
print(" Encoded User & Song IDs:")
print(df[['user_id', 'song_id', 'track_name', 'artist']].head())
sys.stdout.flush()


#  Define Music Recommendation Model BEFORE It’s Used
class MusicRecommender(nn.Module):
    def __init__(self, num_users, num_songs, embedding_dim):
        super(MusicRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)  # Predict score

    def forward(self, user, song):
        user_vec = self.user_embedding(user)
        song_vec = self.song_embedding(song)
        interaction = user_vec * song_vec
        return self.fc(interaction).squeeze()


#  Initialize and Load the Trained Model
num_users = len(df['user_id'].unique())
num_songs = len(df['song_id'].unique())
embedding_dim = 50

model = MusicRecommender(num_users, num_songs, embedding_dim)  # Initialize model

#  Load trained weights if the model has been trained
try:
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set to evaluation mode
    print(" Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Model file not found! Training a new model...")


#  Define Flask Routes
@app.route('/')
def home():
    return "Flask server is running and connected to MongoDB!"

@app.route('/songs', methods=['GET'])
def get_songs():
    songs = list(collection.find({}, {"_id": 0}))  # Get all songs, exclude _id field
    return jsonify(songs)

@app.route('/recommend', methods=['GET'])
def recommend():
    print(" /recommend API called")  # Debug print
    
    # Load trained model
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set model to evaluation mode
    
    test_user = torch.LongTensor([df['user_id'].iloc[0]])  # Get a valid user ID
    song_ids = torch.LongTensor(df['song_id'].unique())  # Get all song IDs
    
    # Get predictions for all songs
    scores = model(test_user, song_ids)

    #  Ensure `scores` is a 1D tensor before sorting
    if scores.dim() == 0:  
        scores = scores.unsqueeze(0)  # Convert to a 1D tensor

    # Get top 5 recommended song indices
    top_songs = scores.argsort(descending=True)[:5]

    recommendations = []
    for idx in top_songs:
        song_data = df[df['song_id'] == song_ids[idx].item()].iloc[0]
        recommendations.append({
            "track_name": song_data['track_name'],
            "artist": song_data['artist'],
            "score": scores[idx].item()
        })

    return jsonify({"recommendations": recommendations})


#  Training the model
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(user_tensor, song_tensor)
    loss = criterion(predictions.view(-1), popularity_tensor)  # Ensure correct shape
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

#  Save the model after training
torch.save(model.state_dict(), 'model.pth')
print(" Model saved successfully!")


if __name__ == '__main__':
    app.run(debug=True)
