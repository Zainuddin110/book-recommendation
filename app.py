import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from urllib.parse import urlparse, parse_qs

# Load the book dataset (Update the file path if necessary)
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

# Function to get book recommendations based on main category, sub category, or language
def get_recommendations(query, cosine_sim=cosine_sim):
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

        for idx in exact_match_indices:
            sim_scores.append((idx, 1.5))  # Boost exact matches

        sim_scores = sorted(list(set(sim_scores)), key=lambda x: x[1], reverse=True)[:50]
        book_indices = [i[0] for i in sim_scores]

        top_results = books_df.iloc[book_indices]
        return top_results[['BookNameAndCode', 'Main category', 'Sub Category', 'Language', 'Book Link']]
    else:
        return "Sorry, no book found matching that query. Please try another."

# Streamlit App UI and Logic
def book_recommendation_system():
    st.title('Book Recommendation System')

    # Get the query parameter from the URL
    query_params = st.experimental_get_query_params()
    query = query_params.get("query", [None])[0]  # Get the 'query' parameter, default to None

    if query:
        st.write(f"Looking for recommendations related to: **{query}**")

        # Call the recommendation function
        recommendations = get_recommendations(query)

        # Display recommendations
        display_recommendations(recommendations)

    else:
        st.write("Please enter a search term in the URL (e.g., ?query=Fiction)")

# Function to display the recommendations in a table with clickable links
def display_recommendations(recommendations):
    if isinstance(recommendations, pd.DataFrame):
        entries_per_page = st.selectbox('Select number of entries to display:', options=[10, 25, 50], index=0)
        limited_recommendations = recommendations.head(entries_per_page)

        if 'Book Link' in recommendations.columns:
            recommendations['Book Link'] = recommendations['Book Link'].apply(
                lambda link: f'<a href="{link}" target="_blank">View Book</a>' if pd.notnull(link) else ''
            )

        st.write(recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write(recommendations)

if __name__ == "__main__":
    book_recommendation_system()
