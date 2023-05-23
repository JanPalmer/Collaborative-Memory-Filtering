import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Based on https://towardsdatascience.com/how-does-collaborative-filtering-work-da56ea94e331

# See a user's preferences
def because_user_liked(user_item_m, movies, ratings, user):
    ix_user_seen = user_item_m.loc[user]>0.
    seen_by_user = user_item_m.columns[ix_user_seen]
    recommendations = (seen_by_user.to_frame()
                 .reset_index(drop=True)
                 .merge(movies)
                 .assign(userId=user)
                 .merge(ratings[ratings.userId.eq(user)])
                 .sort_values('rating', ascending=False).head(10))
    recommendations = recommendations[["movieId", "title", "rating", "genres"]]
    return (recommendations)

# The algorithms
class CfRec():
    def __init__(self, M, X, items, k=20, top_n=10):
        self.X = X
        self.M = M
        self.k = k
        self.top_n = top_n
        self.items = items
        
    def recommend_user_based(self, user):
        ix = self.M.index.get_loc(user)
        # obtain the indices of the top k most similar users
        most_similar = self.X.iloc[ix].nlargest(self.k).index.tolist()
        # Obtain the mean ratings of those users for all movies
        rec_items = self.M.loc[most_similar].mean(0).sort_values(ascending=False)
        # Discard already seen movies
        # already seen movies
        seen_mask = self.M.loc[ix].gt(0)
        seen = seen_mask.index[seen_mask].tolist()
        rec_items = rec_items.drop(seen).head(self.top_n)
        # return recommendations - top similar users rated movies
        recs = (rec_items.index.to_frame()
                                .reset_index(drop=True)
                                .merge(self.items))
        print(f"Recommendations for User {user}")
        print(recs.loc[:, ["title", "genres"]])
        return recs

    def recommend_item_based(self, item):
        # get index of movie        
        ix = self.M.columns.get_loc(item)
        # print(ix)
        liked = self.items.loc[ix, ["title", "genres"]]
        print(f"Because you liked {liked['title']}, {liked['genres']}, we'd recommend you to watch:")
        
        # obtain the indices of the top k most similar items
        most_similar = self.X.iloc[ix].nlargest(self.top_n).index
        #print(self.X.iloc[ix].nlargest(self.top_n))
        recs = self.items.iloc[most_similar]
        print(recs.loc[:, ["title", "genres"]])
        return (recs)

# preparing the ratings table
ratings = pd.read_csv("../ml-latest-small/ratings.csv", sep=",", header=0)
movies = pd.read_csv("../ml-latest-small/movies.csv", sep=",", header=0)
ratings.head()
ratings_matrix = pd.pivot_table(ratings, values='rating', index='userId', columns='movieId')
normalized_ratings_matrix = ratings_matrix.subtract(ratings_matrix.mean(axis=1), axis=0).fillna(0)
#normalized_ratings_matrix2 = ratings_matrix.subtract(2.5, axis=0).fillna(0)
normalized_ratings_matrix2 = ratings_matrix.fillna(0)


# User similarity matrix
similarity_user = cosine_similarity(normalized_ratings_matrix)
np.fill_diagonal(similarity_user, 0)
similarity_user_pd = pd.DataFrame(similarity_user, columns = similarity_user.dtype.names)
similarity_user_pd.index.name = 'userId'
similarity_user_pd.columns.name = 'userId'

# Movies similarity matrix
normalized_ratings_transposed = normalized_ratings_matrix2.T
similarity_movie = cosine_similarity(normalized_ratings_transposed)
np.fill_diagonal(similarity_movie, 0)
similarity_movie_pd = pd.DataFrame(similarity_movie, columns = similarity_movie.dtype.names)
similarity_movie_pd.index.name = 'movieId'
similarity_movie_pd.columns.name = 'movieId'

# Testing, User
rec = CfRec(normalized_ratings_matrix, similarity_user_pd, movies)
userId = 83
print(f"User {userId}, because you liked:")
print(because_user_liked(normalized_ratings_matrix, movies, ratings, userId).loc[:, ["title", "genres"]])
rec.recommend_user_based(userId)

# Testing, Movies
# Różne wyniki rekomendacji zależnie od znormalizowania macierzy ocen
# IMO macierz bez odejmowania srednich ocen użytkowników daje lepsze rekomendacje
rec = CfRec(normalized_ratings_matrix2, similarity_movie_pd, movies)
# Jurassic Park - 480
movieId = 480
recommendations = rec.recommend_item_based(movieId)
# ix = movies.loc(movieId)
# most_similar = similarity_movie_pd.iloc[ix].nlargest(10).tolist()
# most_similar = similarity_movie_pd.iloc[ix].nlargest(10).index.tolist()
# movies.iloc[most_similar]

# ix = normalized_ratings_matrix2.columns.get_loc(movieId)
# print(ix)
# liked = movies.loc[ix, ["title", "genres"]]
# print(liked)
# print(f"Because you liked {liked['title']}, {liked['genres']}, we'd recommend you to watch:")
