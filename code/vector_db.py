import pandas as pd
import numpy as np
from fastembed import TextEmbedding
from chromadb import PersistentClient
from chromadb.api.types import Documents, Embeddings, IDs
from datetime import datetime
from typing import List
from collections import Counter

class RecommenderSystem:
    """
    A recommender system that uses ChromaDB and FastEmbed to generate product recommendations
    based on user purchase history and product embeddings.
    """

    def __init__(self, data_path: str, chroma_db_path: str = "./chromadb"):
        """
        Initializes the recommender system.

        Parameters:
        - data_path (str): Path to the parquet file containing product data with embeddings.
        - chroma_db_path (str): Path where ChromaDB will store data persistently.
        """
        self.data_path = data_path
        self.chroma_db_path = chroma_db_path
        self.embedding_model = TextEmbedding()
        self.df = None
        self.collection = None

    def load_data(self):
        """
        Loads the product data from the parquet file.
        """
        self.df = pd.read_parquet(self.data_path)
        print(f"Data loaded. Number of products: {len(self.df)}")

    def initialize_chromadb(self):
        """
        Initializes the persistent ChromaDB client and creates or loads a collection.
        """
        # Initialize the persistent ChromaDB client
        self.chroma_client = PersistentClient(path=self.chroma_db_path)
        # Check if the collection already exists
        try:
            self.collection = self.chroma_client.get_collection(name="recommendation_sys")
            print("Existing ChromaDB collection loaded.")
        except ValueError:
            # Create a new collection if it doesn't exist
            self.collection = self.chroma_client.create_collection(name="recommendation_sys")
            print("New ChromaDB collection created.")

    def populate_chromadb(self, start_index: int = 0, end_index: int = None):
        """
        Populates the ChromaDB collection with product descriptions and embeddings.

        Parameters:
        - start_index (int): Starting index of the data to add.
        - end_index (int): Ending index of the data to add.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() before populating ChromaDB.")

        if end_index is None:
            end_index = len(self.df)

        documents = self.df["product_description"].values.tolist()[start_index:end_index]
        embeddings = self.df["embeddings"].values.tolist()[start_index:end_index]
        ids = self.df["parent_asin"].values.tolist()[start_index:end_index]

        # Convert data to the appropriate types
        documents = Documents(documents)
        embeddings = Embeddings(embeddings)
        ids = IDs(ids)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )
        print(f"ChromaDB collection populated with {len(documents)} items.")

    def get_user_embeddings(self, user_products: List[str]) -> np.ndarray:
        """
        Calculates the aggregated embedding for a user's purchased products.

        Parameters:
        - user_products (List[str]): List of product IDs (parent_asin) purchased by the user.

        Returns:
        - np.ndarray: Aggregated embedding vector for the user.
        """
        embeddings = self.df[self.df["parent_asin"].isin(user_products)]["embeddings"].values

        if len(embeddings) == 0:
            return None

        # Stack embeddings to form a 2D array
        embeddings_stack = np.stack(embeddings)
        # Aggregate embeddings by averaging
        aggregated_embedding = np.mean(embeddings_stack, axis=0)
        return aggregated_embedding

    def recommend_products(self, user_embedding: np.ndarray, n_results: int = 10) -> List[str]:
        """
        Recommends products based on the user's aggregated embedding.

        Parameters:
        - user_embedding (np.ndarray): The aggregated embedding vector for the user.
        - n_results (int): Number of recommendations to return.

        Returns:
        - List[str]: List of recommended product IDs (parent_asin).
        """
        if user_embedding is None:
            return []

        results = self.collection.query(
            query_embeddings=[user_embedding],
            n_results=n_results,
        )

        # Flatten the list of IDs returned
        recommended_ids = [item for sublist in results["ids"] for item in sublist]
        return recommended_ids

    def calculate_similarity(self, vec1: List[str], vec2: List[str]) -> float:
        """
        Calculates the average cosine similarity between two sets of product embeddings.

        Parameters:
        - vec1 (List[str]): List of product IDs.
        - vec2 (List[str]): List of product IDs.

        Returns:
        - float: Average cosine similarity.
        """
        values = []
        for id1 in vec1:
            emb1 = self.df[self.df["parent_asin"] == id1]["embeddings"]
            if len(emb1) == 0:
                continue
            emb1 = emb1.iloc[0]

            for id2 in vec2:
                emb2 = self.df[self.df["parent_asin"] == id2]["embeddings"]
                if len(emb2) == 0:
                    continue
                emb2 = emb2.iloc[0]

                # Compute cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                values.append(similarity)

        return np.mean(values) if values else 0.0

    def run_recommendation(self, user_purchase_history: pd.DataFrame, test_data: pd.DataFrame):
        """
        Runs the recommendation process for each user and evaluates the recommendations.

        Parameters:
        - user_purchase_history (pd.DataFrame): DataFrame containing users' purchase history.
        - test_data (pd.DataFrame): DataFrame containing users' purchases in the test period.
        """
        users = user_purchase_history["user_id"].unique()
        for user_id in users:
            # Get the products the user purchased in the training period
            train_products = user_purchase_history[user_purchase_history["user_id"] == user_id]["parent_asin"].unique()

            if len(train_products) == 0:
                continue

            # Get the aggregated embedding for the user
            user_embedding = self.get_user_embeddings(train_products)

            if user_embedding is None:
                continue

            # Get recommendations
            recommended_ids = self.recommend_products(user_embedding, n_results=10)

            # Get the products the user purchased in the test period
            test_products = test_data[test_data["user_id"] == user_id]["parent_asin"].unique()

            # Calculate similarity metrics
            train_similarity = self.calculate_similarity(recommended_ids, train_products)
            test_similarity = self.calculate_similarity(recommended_ids, test_products)
            train_test_similarity = self.calculate_similarity(train_products, test_products)

            # Output the results
            print(f"User ID: {user_id}")
            print(f"Train Recommendation Similarity: {train_similarity:.4f}")
            print(f"Test Recommendation Similarity: {test_similarity:.4f}")
            print(f"Train-Test Similarity: {train_test_similarity:.4f}")
            print(f"Recommendations overlap with test purchases: {any(pid in test_products for pid in recommended_ids)}")
            print(f"Number of training products: {len(train_products)}")
            print(f"Number of test products: {len(test_products)}\n")

    def recommend_for_text(self, text: str, n_results: int = 10) -> List[str]:
        """
        Recommends products based on a textual description.

        Parameters:
        - text (str): Input text to generate embedding from.
        - n_results (int): Number of recommendations to return.

        Returns:
        - List[str]: List of recommended product IDs (parent_asin).
        """
        # Generate embedding for the input text
        query_embedding = next(self.embedding_model.embed([text]))
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        # Flatten the list of IDs returned
        recommended_ids = [item for sublist in results["ids"] for item in sublist]
        return recommended_ids

# Usage Example
if __name__ == "__main__":
    # Initialize the recommender system
    recommender = RecommenderSystem(
        data_path="combined_with_embeddings.parquet",
        chroma_db_path="/Users/abhishek/Documents/db"
    )

    # Load data and initialize ChromaDB
    recommender.load_data()
    recommender.initialize_chromadb()

    # Populate ChromaDB with product data (if not already populated)
    # Uncomment the line below if you need to populate the database
    # recommender.populate_chromadb()

    # Prepare user purchase history data
    # Load your training and test data
    data = pd.read_parquet("filtered_data_2021.parquet")
    data["month"] = data["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month)
    train_data = data[data["month"] < 11]
    test_data = data[data["month"] >= 11]

    # Optionally filter for a specific category (e.g., Lips products)
    # train_data = train_data[train_data['Lips'] == 1]
    # test_data = test_data[test_data['Lips'] == 1]

    # Run the recommendation process
    recommender.run_recommendation(user_purchase_history=train_data, test_data=test_data)

    # Example of recommending products based on a text description
    text_query = """The user is looking for an eyeshadow product with the following requirements:
    - Color and Finish: The user used the product for creating a gradient and appreciated its matte finish.
    - Performance: The user mentioned that the product worked well and did not fade after a few months.
    - Application: The user highlighted that the product is silky, soft, and blendable.
    - Recommendation: The user recommends this product for gradients and blending.
    - Sentiment: The overall sentiment of the review is positive."""
    recommendations = recommender.recommend_for_text(text=text_query, n_results=15)
    print("Recommendations based on text query:")
    print(recommendations)
