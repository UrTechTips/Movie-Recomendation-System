import pandas as pd
import numpy as np

tmdb_df = pd.read_csv("./TMDB_movie_dataset_v11.csv")

movies_path = "./ml-latest-small/movies.csv"
ratings_path = "./ml-latest-small/ratings.csv"
tags_path = "./ml-latest-small/tags.csv"
links_path = "./ml-latest-small/links.csv"

movies_df = pd.read_csv(movies_path)
links_df = pd.read_csv(links_path)
ratings_df = pd.read_csv(ratings_path)
tags_df = pd.read_csv(tags_path)

movies_df = pd.merge(movies_df, links_df, on="movieId", how="inner")
movies_df = pd.merge(movies_df, tmdb_df, left_on="tmdbId", right_on="id", how="inner")
movies_df.drop(columns=["title_x", "genres_x", "imdbId", "tmdbId", "backdrop_path", "revenue", "homepage", 'status', 'release_date', 'original_title', 'tagline', 'production_companies', 'production_countries', "adult", "spoken_languages", "keywords", "poster_path", "imdb_id", "budget"], inplace=True)

movies_df.rename(columns={"title_y": "title", "genres_y": "genres"}, inplace=True)

movies_df['runtime_buckets'] = pd.cut(
    movies_df['runtime'],
    bins=[-np.inf, 45, 120, np.inf],
    labels=['short', 'medium', 'long']
)
movies_df['popularity_norm'] = (
    (movies_df['popularity'] - movies_df['popularity'].min()) / 
    (movies_df['popularity'].max() - movies_df['popularity'].min())
)

movies_df['vote_average_norm'] = (
    (movies_df['vote_average'] - movies_df['vote_average'].min()) / 
    (movies_df['vote_average'].max() - movies_df['vote_average'].min())
)
movies_df['vote_count_norm'] = (
    (movies_df['vote_count'] - movies_df['vote_count'].min()) / 
    (movies_df['vote_count'].max() - movies_df['vote_count'].min())
)
movies_df['overview_length'] = movies_df['overview'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

movies_df.to_csv("merged_movies_dataset.csv", index=False)