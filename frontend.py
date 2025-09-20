import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, jsonify
from flask_cors import CORS

# Your API Key and Base URL
# NOTE: In a real-world app, you should not hardcode API keys.
API_KEY = "363a7a5daa3b3122de487e794b02482a"
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/upcoming"
TMDB_FIND_URL = "https://api.themoviedb.org/3/find"


class MovieScorePredictor:
    def __init__(self):
        self.model = None
        self.features = None

    def fetch_data_from_api(self, api_key, num_pages=5):
        """
        Fetches movie data from the TMDb API and returns a DataFrame.
        """
        all_movies = []
        for page in range(1, num_pages + 1):
            params = {
                'api_key': api_key,
                'language': 'en-US',
                'page': page
            }
            try:
                response = requests.get(TMDB_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                all_movies.extend(data['results'])
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data from API: {e}")
                return None

        df = pd.DataFrame(all_movies)
        required_columns = ['vote_average', 'vote_count', 'popularity', 'id']
        if not all(col in df.columns for col in required_columns):
            print("Required columns not found in API data. Returning None.")
            return None

        df = df[df['vote_count'] > 0].copy()
        df = df[required_columns].copy()
        return df

    def train(self, data_path=None, data_df=None, target_column='vote_average'):
        """
        Trains the model using either a local CSV or a DataFrame.
        """
        if data_path:
            data = pd.read_csv(data_path)
        elif data_df is not None:
            data = data_df
        else:
            raise ValueError("Either data_path or data_df must be provided.")

        self.features = [col for col in data.columns if col != target_column]
        if not self.features:
            print("No features found to train the model. Check your data.")
            return

        X = data[self.features]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 8, 10]
        }

        base_model = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)

        print("Starting hyperparameter tuning with GridSearchCV...")
        grid_search.fit(X_train, y_train)
        print("Grid search completed.")

        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Model trained successfully! RMSE: {rmse:.4f}")

    def predict(self, new_movie_features):
        new_data = pd.DataFrame([new_movie_features], columns=self.features)
        return self.model.predict(new_data)[0]


# Initialize the Flask application
app = Flask(__name__)
# Add CORS support to allow cross-origin requests
CORS(app)

# Initialize and train the model once when the server starts
predictor = MovieScorePredictor()
api_data = predictor.fetch_data_from_api(API_KEY, num_pages=5)
if api_data is not None:
    api_data = api_data.rename(columns={'id': 'movie_id'})
    predictor.train(data_df=api_data.drop(columns=['movie_id']), target_column='vote_average')
else:
    print("Could not fetch data for training. The predictor will not work.")
    predictor = None


@app.route('/predict', methods=['POST'])
def predict_score():
    """
    Receives an IMDb link and returns a predicted score.
    """
    if not predictor or not predictor.model:
        return jsonify({'error': 'Model not trained or data not available.'}), 500

    data = request.get_json()
    imdb_link = data.get('imdb_link')
    if not imdb_link:
        return jsonify({'error': 'No IMDb link provided.'}), 400

    imdb_id_match = imdb_link.find("tt")
    if imdb_id_match == -1:
        return jsonify({'error': 'Invalid IMDb link format.'}), 400
    imdb_id = imdb_link[imdb_id_match:]

    # Use TMDb's find API to get movie details from the IMDb ID
    try:
        response = requests.get(
            f"{TMDB_FIND_URL}/{imdb_id}",
            params={'api_key': API_KEY, 'external_source': 'imdb_id'}
        )
        response.raise_for_status()
        tmdb_data = response.json()

        movie_results = tmdb_data.get('movie_results', [])
        if not movie_results:
            return jsonify({'error': 'Movie not found on TMDb or is not a movie.'}), 404

        # Use the first movie result
        movie_info = movie_results[0]

        # Check if vote_count and popularity exist and are valid for the model
        vote_count = movie_info.get('vote_count')
        popularity = movie_info.get('popularity')

        if vote_count is None or popularity is None:
            return jsonify({'error': 'Required features (vote_count, popularity) not available for this movie.'}), 404

        new_movie_features = {
            'vote_count': vote_count,
            'popularity': popularity
        }

        # Predict the score
        predicted_score = predictor.predict(new_movie_features)

        return jsonify({
            'predicted_score': predicted_score
        })

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return jsonify({'error': 'Failed to fetch movie data from TMDb API.'}), 500
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500


if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True, port=5000)
