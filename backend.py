# IMPORTANT: You need to install the imdbpy library to run this script.
# Run this command in your terminal: pip install imdbpy
# It is also recommended to install a faster SQL backend for imdbpy: pip install "imdbpy[sql]"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import imdb

# This is the list of unreleased movies with hypothetical data for prediction.
# The `train` method still uses data from IMDb for training the model.
upcoming_movies_data = [
    {
        "title": "The Batman - Part II",
        "vote_count": 8500,
        "popularity": 150.2,
        "vote_average": None  # No actual score available yet
    },
    {
        "title": "TRON: Ares",
        "vote_count": 5200,
        "popularity": 95.5,
        "vote_average": None
    },
    {
        "title": "Blade",
        "vote_count": 6800,
        "popularity": 110.1,
        "vote_average": None
    },
    {
        "title": "The Super Mario Galaxy Movie",
        "vote_count": 12000,
        "popularity": 250.5,
        "vote_average": None
    },
    {
        "title": "Elio",
        "vote_count": 4500,
        "popularity": 85.3,
        "vote_average": None
    },
    {
        "title": "How to Train Your Dragon",
        "vote_count": 9100,
        "popularity": 180.7,
        "vote_average": None
    },
    {
        "title": "F1",
        "vote_count": 7800,
        "popularity": 135.9,
        "vote_average": None
    },
    {
        "title": "Mission: Impossible – The Final Reckoning",
        "vote_count": 11500,
        "popularity": 210.8,
        "vote_average": None
    },
    {
        "title": "The Mandalorian and Grogu",
        "vote_count": 10500,
        "popularity": 220.1,
        "vote_average": None
    },
    {
        "title": "Masters of the Universe",
        "vote_count": 6100,
        "popularity": 105.6,
        "vote_average": None
    },
    {
        "title": "Mortal Kombat II",
        "vote_count": 7300,
        "popularity": 125.4,
        "vote_average": None
    },
    {
        "title": "Toy Story 5",
        "vote_count": 14000,
        "popularity": 280.9,
        "vote_average": None
    },
    {
        "title": "Supergirl: Woman of Tomorrow",
        "vote_count": 9500,
        "popularity": 190.3,
        "vote_average": None
    }
]


class MovieScorePredictor:
    def __init__(self):
        self.model = None
        self.features = ['vote_count', 'popularity']
        self.data_df = None
        self.rmse = None
        # Initialize the IMDb access object
        self.ia = imdb.IMDb()

    def train(self):
        """
        Fetches a sample of movies from IMDb and trains the model.
        Training a model on a small dataset will result in high RMSE,
        but this demonstrates the concept. For better accuracy, a much larger
        dataset would be needed.
        """
        print("Fetching movie data from IMDb to train the model...")

        # Get the top 250 movies to use as a training set
        top_movies = self.ia.get_top250_movies()

        # We need to fetch details for each movie to get the required features
        movie_data_list = []
        for movie in top_movies[:50]:  # Use a smaller sample for faster startup
            try:
                # Get full movie data
                movie = self.ia.get_movie(movie.getID())
                # Check for required data fields
                if 'votes' in movie and 'rating' in movie and 'popularity' in movie:
                    movie_data_list.append({
                        'title': movie['title'],
                        'vote_count': movie['votes'],
                        'popularity': movie['popularity'],
                        'vote_average': movie['rating']
                    })
            except Exception as e:
                print(f"Skipping movie {movie.get('title', 'N/A')} due to an error: {e}")
                continue

        # Create a pandas DataFrame from the fetched data
        data = pd.DataFrame(movie_data_list)
        self.data_df = data.copy()

        if data.empty or data.shape[0] < 5:
            raise ValueError("Not enough data fetched from IMDb to train the model.")

        X = data[self.features]
        y = data['vote_average']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning to find the best model
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=2)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        print("Model training complete.")

        # Calculate RMSE on the test set
        y_pred = self.model.predict(X_test)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Model trained successfully! RMSE: {self.rmse:.4f}")

    def predict(self, movie_title):
        """
        Predicts the score for a specific movie. It first checks for a hardcoded
        unreleased movie, and if not found, it uses the IMDb API.
        """
        # First, check if the movie is in the unreleased data list
        unreleased_movie_data = next(
            (movie for movie in upcoming_movies_data if movie['title'].lower() == movie_title.lower()),
            None
        )

        if unreleased_movie_data:
            # Use the hardcoded data for prediction
            new_movie_features = {
                'vote_count': unreleased_movie_data['vote_count'],
                'popularity': unreleased_movie_data['popularity']
            }
            new_data = pd.DataFrame([new_movie_features], columns=self.features)
            predicted_score = self.model.predict(new_data)[0]
            return predicted_score, None, 'Prediction based on simulated data for an unreleased movie.'

        else:
            # If not an unreleased movie, search for it on IMDb
            search_results = self.ia.search_movie(movie_title)
            if not search_results:
                raise ValueError(f"Movie '{movie_title}' not found on IMDb.")

            # Get the first search result's ID and fetch its full data
            movie_id = search_results[0].getID()
            movie_data = self.ia.get_movie(movie_id)

            # Ensure the movie has the required features for prediction
            if 'votes' not in movie_data or 'popularity' not in movie_data:
                raise ValueError(f"Required data for prediction (votes, popularity) not available for '{movie_title}'.")

            # Extract features for prediction
            new_movie_features = {
                'vote_count': movie_data['votes'],
                'popularity': movie_data['popularity']
            }

            new_data = pd.DataFrame([new_movie_features], columns=self.features)

            predicted_score = self.model.predict(new_data)[0]

            # Get the actual score if available for comparison
            actual_score = movie_data.get('rating', None)

            return predicted_score, actual_score, 'Prediction based on live IMDb data.'


# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Initialize and train the model once when the server starts
predictor = MovieScorePredictor()
try:
    predictor.train()
except Exception as e:
    print(f"Failed to train the model: {e}")
    predictor.model = None  # Ensure no model is loaded if training fails


@app.route('/predict', methods=['POST'])
def predict_score():
    """
    Receives a movie title from the front end and returns a predicted score.
    """
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
