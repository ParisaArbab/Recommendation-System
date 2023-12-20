# -*- coding: utf-8 -*-
"""
@author: Parisa

This Python script is a movie recommendation system using the LightFM library:

1. Data Loading and Model Creation: It imports necessary libraries, fetches the MovieLens dataset (filtering for movies with a minimum rating of 4.0), and initializes a LightFM model using the 'warp' loss function.

2. Model Training: The LightFM model is trained on the 'train' dataset from MovieLens for 30 epochs using 2 threads.

3. Making Recommendations: A function sample_recommandation is defined to recommend movies. For each specified user, it identifies movies they already like, uses the model to predict new movies they might like, and then prints the top 3 known and recommended movies. Finally, it calls this function for three specific users (IDs 3, 25, 450).
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create Model
model = LightFM(loss= 'warp')
#train Model
model.fit(data['train'], epochs = 30, num_threads=2)


def sample_recommandation(model, data, user_ids):
    
    #number of users and movies in training data
    n_users, n_items = data['train'].shape
    
    #generate recommandations for each user we input
    for user_id in user_ids:
        
        #movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        
        #rank them in order of most liked to least 
        top_items = data['item_labels'][np.argsort(-scores)]
        
        #print out the result
        print("User %s"  % user_id)
        print("    known positives:")
        
        for x in known_positives[:3]:
            print("             %s"   % x)
            
        print("     recommanded:")
        for x in top_items[:3]:
            print("        %s" % x)
            
sample_recommandation(model, data, [3, 25, 450])