import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class MovieScorePredictor:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.features = None

    def train(self, data_path, target_column='movie_score'):
        data = pd.read_csv(data_path)
        self.features = [col for col in data.columns if col != target_column]
        X = data[self.features]
        y = data[target_column]
#hi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # You could also add evaluation logic here

    def predict(self, new_movie_features):
        new_data = pd.DataFrame([new_movie_features], columns=self.features)
        return self.model.predict(new_data)[0]

# --- How to use the class ---
# predictor = MovieScorePredictor()
# predictor.train('your_dataset.csv')
# new_movie = {'director_rank': 2, 'budget': 220, 'genre_code': 1}
# predicted_score = predictor.predict(new_movie)
# print(f"Predicted score: {predicted_score:.2f}")