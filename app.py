import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the book dataset (Update the file path if necessary)
file_path = 'Books_Dataset.xlsx'  # Update this path
books_df = pd.read_excel(file_path)

# Fill NaN values with empty strings to prevent errors in text processing
books_df['Author'] = books_df['Author'].fillna('')
books_df['Category'] = books_df['Category'].fillna('')
books_df['Publisher'] = books_df['Publisher'].fillna('')
books_df['Price Starting With ($)'] = books_df['Price Starting With ($)'].fillna('')
books_df['Publish Date (Year)'] = books_df['Publish Date (Year)'].fillna('')

# Combine relevant features into a single column for comparison
books_df['combined_features'] = (books_df['Author'] + " " + 
                                  books_df['Category'] + " " + 
                                  books_df['Publisher'] + " " + 
                                  books_df['Price Starting With ($)'].astype(str) + " " + 
                                  books_df['Publish Date (Year)'].astype(str))

# Vectorize the text data (convert text to numerical form)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(books_df['combined_features'])

# Compute cosine similarity between books
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get book recommendations based on author, category, publisher, price, or publish year
def get_recommendations(query, cosine_sim=cosine_sim):
    # First, prioritize exact matches by book title
    exact_matches = books_df[books_df['Title'].str.contains(query, case=False, na=False)]
    
    # If there are no exact matches, proceed with the general search
    mask = (books_df['Title'].str.contains(query, case=False, na=False) |
            books_df['Author'].str.contains(query, case=False, na=False) |
            books_df['Category'].str.contains(query, case=False, na=False) |
            books_df['Publisher'].str.contains(query, case=False, na=False) |
            books_df['Price Starting With ($)'].str.contains(query, case=False, na=False) |
            books_df['Publish Date (Year)'].str.contains(query, case=False, na=False))
    
    if mask.any():
        # Get the index of the matching book(s)
        indices = books_df[mask].index
        
        # Get pairwise similarity scores of all books with the selected book(s)
        sim_scores = []
        for idx in indices:
            sim_scores += list(enumerate(cosine_sim[idx]))
        
        # Remove duplicates and sort the books based on similarity scores
        sim_scores = sorted(list(set(sim_scores)), key=lambda x: x[1], reverse=True)
        
        # Get the indices of the top 50 most similar books
        sim_scores = sim_scores[:50]
        
        # Get the book indices
        book_indices = [i[0] for i in sim_scores]
        
        # Combine exact matches at the top followed by the similar books
        if not exact_matches.empty:
            top_results = pd.concat([exact_matches, books_df.iloc[book_indices]]).drop_duplicates().head(50)
        else:
            top_results = books_df.iloc[book_indices]
        
        # Return the top similar books
        return top_results[['Title', 'Author', 'Category', 'Publisher', 'Price Starting With ($)', 'Publish Date (Year)', 'Link']]
    else:
        return "Sorry, no book, author, category, publisher, price, or publish year found matching that query. Please try another."

# Streamlit App UI and Logic
def chatbot():
    st.title('Book Recommendation System')

    # Check if there are any query parameters passed from the website
    query_params = st.experimental_get_query_params()

    # If a query parameter exists, pre-fill the input box and show results
    if 'book' in query_params:
        book_query = query_params['book'][0]
        st.write(f"Recommendations for: {book_query}")
        recommendations = get_recommendations(book_query)

        # Display recommendations
        display_recommendations(recommendations)
    else:
        # Input for book title, author, category, publisher, price, or publish year
        user_input = st.text_input("Enter a book title, author, category, publisher, price, or publish year:")

        if user_input:
            # Provide book recommendations based on user input
            st.write(f"Looking for recommendations related to: **{user_input}**")

            # Call the recommendation function
            recommendations = get_recommendations(user_input)

            # Display recommendations
            display_recommendations(recommendations)

# Function to display the recommendations in a table with selectable entries and justified text
def display_recommendations(recommendations):
    if isinstance(recommendations, pd.DataFrame):
        # Select the number of entries to display
        entries_per_page = st.selectbox('Select number of entries to display:', options=[10, 25, 50], index=0)

        # Limit the recommendations to the selected number of entries
        limited_recommendations = recommendations.head(entries_per_page)

        # Modify the DataFrame to include clickable links
        limited_recommendations['Link'] = limited_recommendations['Link'].apply(lambda x: f'<a href="{x}" target="_blank">Listen here</a>')

        # HTML style for justified text in the table
        html_style = """
        <style>
            table {
                width: 100%;
                text-align: justify;
            }
            th, td {
                padding: 10px;
                text-align: justify;
                border-bottom: 1px solid #ddd;
            }
        </style>
        """

        # Display the data frame with clickable links and justified text
        st.markdown(html_style, unsafe_allow_html=True)
        st.write(limited_recommendations[['Title', 'Author', 'Category', 'Publisher', 'Price Starting With ($)', 'Publish Date (Year)', 'Link']].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write(recommendations)

# Run the chatbot
if __name__ == "__main__":
    chatbot()
