import re
import pandas as pd
import numpy as np
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- DISPLAY (UNCHANGED) ----
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)


# =====================================================
# TF-IDF SIMILARITY
# =====================================================
def compute_tfidf(df):
    df['overview'] = df['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=6000, ngram_range=(1,2))
    mat = tfidf.fit_transform(df['overview'])
    sim = cosine_similarity(mat, mat)
    idx = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    return sim, idx


# =====================================================
# LOAD DATA
# =====================================================
def load_dataset(path):

    df = pd.read_csv(path)

    df.fillna({
        'runtime':0,'genres':'','overview':'',
        'popularity_norm':0,'vote_average_norm':0,
        'vote_count_norm':0,'overview_length':0
    }, inplace=True)

    C = df['vote_average_norm'].mean()
    m = df['vote_count_norm'].quantile(0.6)

    df['weighted_rating'] = (
        (df['vote_count_norm']/(df['vote_count_norm']+m))*df['vote_average_norm']
        + (m/(df['vote_count_norm']+m))*C
    )

    df['story_quality'] = np.log1p(df['overview_length'])
    df['story_quality'] /= df['story_quality'].max() if df['story_quality'].max()!=0 else 1

    df['rating_confidence'] = np.sqrt(df['vote_count_norm'])

    sim, idx = compute_tfidf(df)
    return df, sim, idx


# =====================================================
# MULTI MOVIE EXTRACTOR
# =====================================================
def extract_movies_from_text(text):

    text = text.lower()

    patterns = [
        r'liked\s+(.*)',
        r'loved\s+(.*)',
        r'watched\s+(.*)',
        r'story of\s+(.*)',
        r'about\s+(.*)',
        r'details of\s+(.*)',
        r'summary of\s+(.*)'
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            part = m.group(1)
            movies = re.split(r',| and ', part)
            return [x.strip() for x in movies if len(x.strip()) > 1]

    return []


# =====================================================
# INTENT
# =====================================================
def is_details_query(text):
    return any(k in text.lower() for k in ["story","plot","details","about","summary","tell me about"])


# =====================================================
# INPUT PARSER
# =====================================================
def parse_input(text):

    t = text.lower()

    time = 120
    if re.search(r'(\d+)\s*hour', t):
        time = int(re.search(r'(\d+)\s*hour', t).group(1))*60
    elif re.search(r'(\d+)\s*min', t):
        time = int(re.search(r'(\d+)\s*min', t).group(1))

    k = int(re.search(r'\b(\d+)\b', t).group(1)) if re.search(r'\b(\d+)\b', t) else 5
    k = min(k, 20)

    movies = extract_movies_from_text(text)
    print(f"Extracted movies: {movies}")

    genres = {'Fantasy', 'Western', 'Romance', 'Crime', 'Comedy', 'TV Movie', 'Adventure', 'Documentary', 'Mystery', 'Family', 'War', 'Thriller', 'Music', 'Drama', 'Animation', 'Science Fiction', 'Action', 'History', 'Horror'}

    genre = next((g.capitalize() for g in genres if g.lower() in t), None)

    lang = "en" if "english" in t else "hi" if "hindi" in t else None

    return time, k, genre, movies, lang


# =====================================================
# SHOW DETAILS
# =====================================================
def show_movie_details(df, movie):

    match = get_close_matches(movie.lower(), df['title'].str.lower(), n=1, cutoff=0.6)

    if not match:
        print("🤖 Sorry, I couldn't find that movie.")
        return

    row = df[df['title'].str.lower()==match[0]].iloc[0]

    print("\n🎬 Movie Details\n")
    print(f"Title: {row['title']}")
    print(f"Genres: {row['genres']}")
    print(f"Runtime: {row['runtime']} minutes")
    print(f"Rating: {row['vote_average_norm']}")
    print(f"Popularity: {row['popularity_norm']}")
    print("\n📖 Story:\n")
    print(row['overview'])
    print("\n"+"-"*80)


# =====================================================
# HYBRID RECOMMENDER (MULTI MOVIE AWARE)
# =====================================================
def recommend(df, sim, idx, time_limit, top_k, genre, movies, lang):

    rec_df = df[df['runtime'] <= time_limit].copy()
    rec_df['runtime_comfort'] = 1 - abs(rec_df['runtime']-time_limit)/time_limit

    if genre:
        rec_df['genre_match'] = rec_df['genres'].str.contains(genre, case=False).astype(int)
    else:
        rec_df['genre_match'] = 0

    if lang:
        rec_df['lang_match'] = (rec_df['original_language']==lang).astype(int)
    else:
        rec_df['lang_match'] = 0

    # ---- MULTI SIMILARITY PROFILE ----
    sim_vectors = []

    for m in movies:
        close = get_close_matches(m.lower(), idx.index, n=1, cutoff=0.6)
        if close:
            sim_vectors.append(sim[idx[close[0]]])

    if sim_vectors:
        combined_sim = np.mean(sim_vectors, axis=0)
        rec_df['similarity'] = combined_sim[rec_df.index]
    else:
        rec_df['similarity'] = 0

    if rec_df['similarity'].max() > 0:
        rec_df['similarity'] = (rec_df['similarity']-rec_df['similarity'].min()) / \
                               (rec_df['similarity'].max()-rec_df['similarity'].min())

    rec_df['score'] = (
        0.25*rec_df['weighted_rating'] +
        0.2*rec_df['similarity'] +
        0.1*rec_df['popularity_norm'] +   
        0.07*rec_df['story_quality'] +
        0.08*rec_df['runtime_comfort'] +
        0.05*rec_df['rating_confidence'] +
        0.15*rec_df['lang_match'] + 
        0.15*rec_df['genre_match']
    )

    for m in movies:
        rec_df = rec_df[rec_df['title'].str.lower() != m.lower()]

    return rec_df[['title','genres','runtime',
                   'similarity','weighted_rating',
                   'runtime_comfort','rating_confidence','score']]\
        .sort_values(by='score',ascending=False)\
        .head(top_k)


# =====================================================
# CHAT LOOP
# =====================================================
if __name__=="__main__":

    df, sim, idx = load_dataset("merged_movies_dataset.csv")

    print("\n🎬 Hybrid Movie Assistant Ready!")
    print("Examples:")
    print(" i watched john wick, godfather, shawshank redemption")
    print(" tell me about inside out\n")

    while True:
        text = input("You: ")

        if text.lower() in ["exit","quit","bye"]:
            break

        time, k, genre, movies, lang = parse_input(text)

        if is_details_query(text) and movies:
            show_movie_details(df, movies[0])
        else:
            recs = recommend(df, sim, idx, time, k, genre, movies, lang)
            print("\n🤖 Recommendations:\n")
            print(recs.to_string(index=False, justify='left'))
            print("\n"+"-"*80)