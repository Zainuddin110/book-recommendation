import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

# Load the book dataset
file_path = 'AWGP_Databases.xlsx'  # Replace with your dataset file path
books_df = pd.read_excel(file_path)

# Fill NaN values with empty strings to prevent errors in text processing
books_df['Main category'] = books_df['Main category'].fillna('')
books_df['Sub Category'] = books_df['Sub Category'].fillna('')
books_df['Language'] = books_df['Language'].fillna('')

# Combine relevant features into a single column for comparison
books_df['combined_features'] = (books_df['Main category'] + " " +
                                  books_df['Sub Category'] + " " +
                                  books_df['Language'])

# Vectorize the text data (convert text to numerical form)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(books_df['combined_features'])

# Compute cosine similarity between books
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# FastAPI application
app = FastAPI()

class Recommendation(BaseModel):
    book_name: str
    main_category: Optional[str] = None
    sub_category: Optional[str] = None
    language: Optional[str] = None
    book_link: Optional[str] = None

@app.get("/")
def home():
    return {"message": "Welcome to the Book Recommendation API"}

@app.get("/recommendations", response_model=List[Recommendation])
def get_recommendations_api(query: str = Query(..., description="Search by book name, category, or language")):
    # Fill NaN values with empty strings
    books_df.fillna('', inplace=True)

    # Create a mask for matching the query across relevant columns
    mask = (
        books_df['Main category'].str.contains(query, case=False, na=False) |
        books_df['Sub Category'].str.contains(query, case=False, na=False) |
        books_df['Language'].str.contains(query, case=False, na=False) |
        books_df['BookNameAndCode'].str.contains(query, case=False, na=False)
    )

    if mask.any():
        indices = books_df[mask].index
        sim_scores = []

        # Calculate similarity scores for matching rows
        for idx in indices:
            sim_scores += list(enumerate(cosine_sim[idx]))

        # Prioritize exact matches by boosting their scores
        exact_match_mask = books_df['BookNameAndCode'].str.contains(query, case=False, na=False)
        exact_match_indices = books_df[exact_match_mask].index

        # Add exact matches with a high score (e.g., 1.5) to prioritize them
        for idx in exact_match_indices:
            sim_scores.append((idx, 1.5))  # Boost exact matches

        # Remove duplicates and sort by similarity score
        sim_scores = sorted(list(set(sim_scores)), key=lambda x: x[1], reverse=True)[:50]
        book_indices = [i[0] for i in sim_scores]

        top_results = books_df.iloc[book_indices]
        recommendations = [
            {
                "book_name": row['BookNameAndCode'],
                "main_category": row['Main category'],
                "sub_category": row['Sub Category'],
                "language": row['Language'],
                "book_link": row['Book Link'],
            }
            for _, row in top_results.iterrows()
        ]
        return recommendations
    else:
        return []
