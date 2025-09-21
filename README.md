# Movie Score Predictor

Movie Score Predictor is a web application that estimates how well an upcoming movie might perform based on factors like popularity, vote count, and other movie features. The goal is to give movie fans an idea of potential ratings before the movie is released. 

The application works by using a machine learning model trained on historical movie data. For movies that already have ratings, the model learns the relationship between features like vote count and popularity and the actual ratings. Then, for unreleased movies, the model predicts a score using the same features, allowing users to see an estimated rating even before the movie hits theaters.

The frontend is interactive, letting users select movies from a list and immediately see predictions. The backend is built with Flask and hosts the ML model, while the predictions are generated using a Gradient Boosting Regressor from scikit-learn. The model is trained on a mix of historical IMDb data and hardcoded upcoming movie information to make the predictions as accurate as possible, and the RMSE (Root Mean Square Error) is displayed to indicate the model’s performance.

This project demonstrates how machine learning can be applied to real-world, fun scenarios, combining predictive analytics with a clean and interactive user interface.

## Developers
- Yaseen Ben  
- Adam Lashnuk  
- Adham
