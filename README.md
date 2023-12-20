# Recommendation-System
This Python script is a movie recommendation system using the LightFM library:

1. Data Loading and Model Creation: It imports necessary libraries, fetches the MovieLens dataset (filtering for movies with a minimum rating of 4.0), and initializes a LightFM model using the 'warp' loss function.

2. Model Training: The LightFM model is trained on the 'train' dataset from MovieLens for 30 epochs using 2 threads.

3. Making Recommendations: A function sample_recommandation is defined to recommend movies. For each specified user, it identifies movies they already like, uses the model to predict new movies they might like, and then prints the top 3 known and recommended movies. Finally, it calls this function for three specific users (IDs 3, 25, 450).
