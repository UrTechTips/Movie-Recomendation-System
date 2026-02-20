import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_tfidf(dataset):
    dataset['overview'] = dataset['overview'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['overview'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(dataset.index, index=dataset['title'].str.lower()).drop_duplicates()
    return cosine_sim, indices

def load_dataset(path):
    try:
        dataset = pd.read_csv(path)
        cosine, indices = compute_tfidf(dataset)
        return dataset, cosine, indices
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

def create_time_fit(available_time_fit, dataset):
    fittable_movies = dataset[dataset['runtime'] <= available_time_fit].copy()
    # Simplified math
    fittable_movies['time_fit'] = fittable_movies['runtime'] / available_time_fit 
    return fittable_movies # Removed redundant sort

def add_genre_score(df, preferred_genre):
    # Vectorized string matching
    df['genre_match'] = df['genres'].str.contains(preferred_genre, case=False, na=False).astype(int) 
    return df

def recomend(dataset, cosine_sim, indices, **kwargs):
    user_input = {
        "available_time_fit": kwargs.get("available_time_fit", 120),
        "top_k": kwargs.get("top_k", 10),
        "preferred_genre": kwargs.get("preferred_genre", "Action"),
        "preferred_movie": kwargs.get("preferred_movie", None)
    }
    
    dataset = create_time_fit(user_input['available_time_fit'], dataset)
    dataset = add_genre_score(dataset, preferred_genre=user_input['preferred_genre'])
    
    if user_input['preferred_movie']:
        if user_input['preferred_movie'].lower() not in indices:
            print(f"Preferred movie '{user_input['preferred_movie']}' not found in dataset. Ignoring similarity feature.")
            dataset['similarity'] = 0
        else:
            idx = indices[user_input['preferred_movie'].lower()]
            print("DEBUG: Similarity scores for '{}': {}".format(user_input['preferred_movie'], cosine_sim[idx]))
            dataset['similarity'] = dataset.index.map(lambda x: cosine_sim[idx][x])
    else:
        dataset['similarity'] = 0

    dataset['score'] = (0.3 * dataset['vote_average_norm']) + \
                        (0.1 * dataset['popularity_norm']) + \
                        (0.1 * dataset['time_fit']) + \
                        (0.1 * dataset['vote_count_norm']) + \
                        (0.1 * dataset['genre_match']) + \
                        (0.3 * dataset['similarity']) 
    
    if user_input['preferred_movie']: # Already watched so score 0
        dataset.loc[dataset['title'].str.lower() == user_input['preferred_movie'].lower(), 'score'] = 0
    
    return dataset[['title', 'genres', 'time_fit', 'similarity', 'score',]]\
           .sort_values(by='score', ascending=False)\
           .head(user_input['top_k'])

def parse_user_input(text):
    time_match = re.search(r'(\d+)\s*(?:minutes|mins|m)', text, re.IGNORECASE)
    available_time = int(time_match.group(1)) if time_match else 120

    k_match = re.search(r'(\d+)\s*(?:\w+\s+)*movies', text, re.IGNORECASE)
    top_k = int(k_match.group(1)) if k_match else 5

    prefered_movie = re.search(r'(?:liked|loved)\s+(?:the\s+)?(["\'])(.*?)(?:\1)', text, re.IGNORECASE)
    prefered_movie = prefered_movie.group(2) if prefered_movie else None

    genres = {'Fantasy', 'Western', 'Romance', 'Crime', 'Comedy', 'TV Movie', 'Adventure', 'Documentary', 'Mystery', 'Family', 'War', 'Thriller', 'Music', 'Drama', 'Animation', 'Science Fiction', 'Action', 'History', 'Horror'}
    preferred_genre = "Action"
    for g in genres:
        if g.lower() in text.lower():
            preferred_genre = g.capitalize()
            break
            
    return available_time, top_k, preferred_genre, prefered_movie

if __name__ == "__main__":
    print("Loading dataset and computing similarities...")
    dataset, cosine_sim, indices = load_dataset("merged_movies_dataset.csv")
    if dataset is not None:
        try:
            # sent = input("Hello! How are you feeling today? \n")
            sent = "Give me 5 movies. I liked 'john wick'"
            available_time, top_k, preferred_genre, preferred_movie = parse_user_input(sent)
            print("DEBUG: Parsed user input - Available Time: {}, Top K: {}, Preferred Genre: {}, Preferred Movie: {}".format(available_time, top_k, preferred_genre, preferred_movie))
            recommendations = recomend(dataset, cosine_sim, indices, available_time_fit=available_time, top_k=top_k, preferred_genre=preferred_genre, preferred_movie=preferred_movie)
            print(recommendations)    
        except ValueError:
            print("Please enter valid numbers for time and recommendations.")