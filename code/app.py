from fastembed import TextEmbedding

import chromadb
import pandas as pd
import numpy as np
import streamlit as st

# Initialize ChromaDB client and get the recommendation system collection
chroma_client = chromadb.PersistentClient(path="/Users/abhishek/Documents/db")
collection = chroma_client.get_collection(name="recommendation_sys")

# Initialize the embedding model
embedding_model = TextEmbedding()

# Load data for images and purchase history
images_data = pd.read_parquet("data/beauty_data_with_images_simple.parquet")
purchase_history = pd.read_parquet("data/purchase_history.parquet")

# Set the Streamlit page configuration to use a wide layout
st.set_page_config(layout="wide")


def get_user_recommendation(user_id: str):
    """
    Generate product recommendations based on a user's purchase history.

    Args:
        user_id (str): The ID of the user.

    Returns:
        tuple: A tuple containing a list of image URLs and a list of titles.
    """
    # Filter the purchase history for the given user and ratings >= 3
    user_purchases = purchase_history[
        (purchase_history["user_id"] == user_id) & (purchase_history["rating"] >= 3)
    ]

    # If the user has no qualifying purchases, return None
    if len(user_purchases) == 0:
        return None, None

    # Get the embeddings of the purchased items
    purchase_embeddings = user_purchases["embeddings"].values

    # Sum the embeddings to create a query embedding
    query_embeddings = np.sum(purchase_embeddings, axis=0).tolist()

    # Query the collection for recommendations based on the query embedding
    results = collection.query(
        query_embeddings=[query_embeddings],
        n_results=5,
    )

    # Get the image URLs and titles for the recommended items
    image_urls, titles = display_images(results["ids"][0])

    return image_urls, titles


def display_images(recom_ids: list):
    """
    Retrieve image URLs and titles for the given product IDs.

    Args:
        recom_ids (list): A list of recommended product IDs.

    Returns:
        tuple: A tuple containing a list of image URLs and a list of titles.
    """
    # Filter the images_data DataFrame for the recommended IDs
    recom = images_data[images_data["parent_asin"].isin(recom_ids)]

    # Get the list of image URLs and titles
    image_urls = recom["images"].values.tolist()
    titles = recom["title"].values.tolist()

    return image_urls, titles


def process_data(text: str, num_images: int):
    """
    Process the user input text to generate product recommendations.

    Args:
        text (str): The search text input by the user.
        num_images (int): The number of images to display.

    Returns:
        tuple: A tuple containing a list of image URLs and a list of titles.
    """
    # Generate embeddings for the input text
    query_embeddings = list(embedding_model.embed([text]))

    # Query the collection for recommendations
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=num_images,
    )

    # Get the image URLs and titles for the recommended items
    image_urls, titles = display_images(results["ids"][0])

    return image_urls, titles


def render_images(image_urls: list, titles: list):
    """
    Render images and titles in a grid layout using Streamlit.

    Args:
        image_urls (list): A list of image URLs.
        titles (list): A list of titles corresponding to the images.
    """
    num_images = len(image_urls)
    # Calculate the number of rows needed (assuming 4 images per row)
    rows = (num_images + 3) // 4

    # Loop over the rows
    for i in range(rows):
        cols = st.columns(4)
        # Loop over each column in the row
        for j in range(4):
            index = i * 4 + j
            if index < num_images:
                url = image_urls[index]
                title = titles[index]
                with cols[j]:
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #ccc; padding: 10px; text-align: center;">
                            <img src="{url}" style="width: 200px; height: 200px; object-fit: cover;">
                            <div>{title}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


def main():
    """
    Main function to run the Streamlit app.
    """
    # Set the title of the app
    st.title("Beauty Recommendations")
    user_id = None

    # User settings in the sidebar
    with st.sidebar:
        st.header("User Settings")
        existing_user = st.checkbox("Are you an existing user?")
        if existing_user:
            user_id = st.text_input("Enter your User ID")
            st.write(f"Welcome back! Your User ID is: {user_id}")
        # Slider to select the number of images to display
        num_images = st.slider(
            "Select number of images to display", min_value=3, max_value=15, value=6
        )

    # If user ID is provided, show recommendations based on purchase history
    if user_id:
        st.header("Based on your previous purchases, here are your recommended products")
        image_urls, titles = get_user_recommendation(user_id)
        if image_urls and titles:
            render_images(image_urls, titles)
        else:
            st.write("No recommendations found for your purchase history.")

    # Input field for the user to search products
    user_input = st.text_input("Find products based on search")

    # When the user clicks the 'Submit' button
    if st.button("Submit"):
        st.header("Based on your search, here are your recommended products")
        image_urls, titles = process_data(user_input, num_images)
        render_images(image_urls, titles)


if __name__ == "__main__":
    main()
