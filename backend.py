import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# This is the list of movies with training examples and unreleased ones
upcoming_movies_data = [
    {"title": "Sample Movie 1", "vote_count": 5000, "popularity": 120.0, "vote_average": 7.5},
    {"title": "Sample Movie 2", "vote_count": 8000, "popularity": 200.0, "vote_average": 8.2},
    {"title": "Sample Movie 3", "vote_count": 3000, "popularity": 90.0, "vote_average": 6.8},
    {"title": "Sample Movie 4", "vote_count": 10000, "popularity": 250.0, "vote_average": 8.9},
    {"title": "Sample Movie 5", "vote_count": 7000, "popularity": 150.0, "vote_average": 7.2},

    {"title": "The Batman - Part II", "vote_count": 8500, "popularity": 150.2, "vote_average": None},
    {"title": "TRON: Ares", "vote_count": 5200, "popularity": 95.5, "vote_average": None},
    {"title": "Blade", "vote_count": 6800, "popularity": 110.1, "vote_average": None},
    {"title": "The Super Mario Galaxy Movie", "vote_count": 12000, "popularity": 250.5, "vote_average": None},
    {"title": "Elio", "vote_count": 4500, "popularity": 85.3, "vote_average": None},
    {"title": "How to Train Your Dragon", "vote_count": 9100, "popularity": 180.7, "vote_average": None},
    {"title": "F1", "vote_count": 7800, "popularity": 135.9, "vote_average": None},
    {"title": "Mission: Impossible – The Final Reckoning", "vote_count": 11500, "popularity": 210.8, "vote_average": None},
    {"title": "The Mandalorian and Grogu", "vote_count": 10500, "popularity": 220.1, "vote_average": None},
    {"title": "Masters of the Universe", "vote_count": 6100, "popularity": 105.6, "vote_average": None},
    {"title": "Mortal Kombat II", "vote_count": 7300, "popularity": 125.4, "vote_average": None},
    {"title": "Toy Story 5", "vote_count": 14000, "popularity": 280.9, "vote_average": None},
    {"title": "Supergirl: Woman of Tomorrow", "vote_count": 9500, "popularity": 190.3, "vote_average": None},
    {"title": "Avengers: Secret Wars", "vote_count": 11000, "popularity": 300.0, "vote_average": None},
    {"title": "Deadpool & Wolverine", "vote_count": 6700, "popularity": 150.0, "vote_average": None},
    {"title": "Joker: Folie à Deux", "vote_count": 9800, "popularity": 210.0, "vote_average": None},
    {"title": "Fantastic Four", "vote_count": 7200, "popularity": 140.0, "vote_average": None},
    {"title": "Spider-Man: Beyond the Spider-Verse", "vote_count": 10500, "popularity": 230.0, "vote_average": None},
    {"title": "Guardians of the Galaxy Vol. 4", "vote_count": 8800, "popularity": 180.0, "vote_average": None},
    {"title": "Doctor Strange 3", "vote_count": 9300, "popularity": 175.0, "vote_average": None},
    {"title": "Captain America: Brave New World", "vote_count": 8900, "popularity": 160.0, "vote_average": None},
    {"title": "Thunderbolts", "vote_count": 6400, "popularity": 120.0, "vote_average": None},
    {"title": "Shang-Chi 2", "vote_count": 7100, "popularity": 145.0, "vote_average": None},
    {"title": "Eternals 2", "vote_count": 5500, "popularity": 110.0, "vote_average": None},
    {"title": "Ironheart", "vote_count": 4800, "popularity": 100.0, "vote_average": None}
]



class MovieScorePredictor:
    def __init__(self):
        self.model = None
        self.features = ['vote_count', 'popularity']
        self.rmse = None

    def train(self):
        data = pd.DataFrame(upcoming_movies_data)
        data = data.dropna(subset=["vote_average"])
        if data.empty:
            raise ValueError("No labeled data available for training.")

        X = data[self.features]
        y = data['vote_average']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = GradientBoostingRegressor(random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Model trained successfully! RMSE: {self.rmse:.4f}")

    def predict(self, movie_title):
        # First, check if the movie is in the hardcoded data
        movie_data = next(
            (movie for movie in upcoming_movies_data if movie['title'].lower() == movie_title.lower()),
            None
        )
        if not movie_data:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        new_movie_features = {
            'vote_count': movie_data['vote_count'],
            'popularity': movie_data['popularity']
        }

        new_data = pd.DataFrame([new_movie_features], columns=self.features)
        predicted_score = self.model.predict(new_data)[0]

        return predicted_score, movie_data.get('vote_average', None), 'Prediction based on static dataset.'


#Flask application
app = Flask(__name__)
CORS(app)

predictor = MovieScorePredictor()
try:
    predictor.train()
except Exception as e:
    print(f"Failed to train the model: {e}")
    predictor.model = None


@app.route('/predict', methods=['POST'])
def predict_score():
    data = request.get_json()
    movie_title = data.get('movie_title')

    if not movie_title:
        return jsonify({'error': 'No movie title provided.'}), 400

    if not predictor.model:
        return jsonify({'error': 'Model is not trained. Please check the server logs.'}), 503

    try:
        predicted_score, actual_score, info_message = predictor.predict(movie_title)

        response_data = {
            'predicted_score': float(predicted_score),
            'info': info_message,
            'rmse': float(predictor.rmse)
        }
        if actual_score is not None:
            response_data['actual_score'] = float(actual_score)

        return jsonify(response_data)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
