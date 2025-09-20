import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Your API Key and Base URL
API_KEY = "363a7a5daa3b3122de487e794b02482a"
BASE_URL = "https://api.themoviedb.org/3/movie/upcoming"


class MovieScorePredictor:
    def __init__(self):
        # We don't initialize the model here anymore because GridSearchCV will do it
        # self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
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
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
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

        # Filter out movies with 0 vote count as they often have a vote average of 0.0
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

        # 1. Set up the parameter grid for Grid Search
        param_grid = {
            'n_estimators': [25 ,50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 8, 10]
        }

        # 2. Initialize a base Gradient Boosting model
        base_model = GradientBoostingRegressor(random_state=42)

        # 3. Initialize GridSearchCV with the model and the parameter grid
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)

        # 4. Run the grid search to find the best model
        print("Starting hyperparameter tuning with GridSearchCV...")
        grid_search.fit(X_train, y_train)
        print("Grid search completed.")

        # 5. Get the best model from the search
        self.model = grid_search.best_estimator_

        # 6. Evaluate the best model's performance
        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Model trained successfully! RMSE: {rmse:.4f}")

    def predict(self, new_movie_features):
        new_data = pd.DataFrame([new_movie_features], columns=self.features)
        return self.model.predict(new_data)[0]


# --- How to use the updated class ---
predictor = MovieScorePredictor()

api_data = predictor.fetch_data_from_api(API_KEY, num_pages=5)

if api_data is not None:
    api_data = api_data.rename(columns={'id': 'movie_id'})
    predictor.train(data_df=api_data.drop(columns=['movie_id']), target_column='vote_average')

    new_movie_features = {
        'vote_count': 5000,
        'popularity': 500.0
    }

    predicted_score = predictor.predict(new_movie_features)
    print(f"\nPredicted score for the new movie: {predicted_score:.2f}")