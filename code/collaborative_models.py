import pandas as pd
from surprise import (
    Dataset,
    Reader,
    SVD,
    SVDpp,
    NMF,
    KNNBasic,
    KNNWithMeans,
    KNNBaseline,
    accuracy,
)
from surprise.model_selection import GridSearchCV, train_test_split
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


class SurpriseModelTrainer:
    """
    A class to train and evaluate collaborative filtering models using the Surprise library,
    with support for multiple algorithms and hyperparameter tuning.
    """

    def __init__(self, data_path):
        """
        Initialize the SurpriseModelTrainer with the dataset path.

        Parameters:
        - data_path (str): Path to the parquet data file.
        """
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.data = None
        self.trainset = None
        self.testset = None
        self.algo = None
        self.results = {}
        self.best_params = {}

    def load_data(self):
        """
        Load the dataset from the parquet file and extract the year from the timestamp.
        """
        # Load the data
        self.df = pd.read_parquet(self.data_path)
        # Convert timestamp to datetime and extract the year
        self.df["year"] = self.df["timestamp"].apply(
            lambda x: datetime.fromtimestamp(x / 1000).year
        )

    def split_data(self):
        """
        Split the data into training and testing sets based on the year.
        Training data includes years before 2023, and testing data is from 2023.
        """
        self.train_df = self.df[self.df["year"] < 2023]
        self.test_df = self.df[self.df["year"] == 2023]

    def filter_data(self, min_item_interactions=30, min_user_interactions=20):
        """
        Filter out items and users with fewer interactions than specified thresholds.

        Parameters:
        - min_item_interactions (int): Minimum number of interactions for an item to be included.
        - min_user_interactions (int): Minimum number of interactions for a user to be included.
        """
        # Count item interactions and filter
        item_counts = self.train_df["asin"].value_counts()
        item_list = item_counts[item_counts > min_item_interactions].index

        # Count user interactions and filter
        user_counts = self.train_df["user_id"].value_counts()
        user_list = user_counts[user_counts > min_user_interactions].index

        # Filter the training data based on the lists
        self.train_df = self.train_df[
            (self.train_df["user_id"].isin(user_list))
            & (self.train_df["asin"].isin(item_list))
        ]

        # Also filter the test data to include only known users and items
        self.test_df = self.test_df[
            (self.test_df["user_id"].isin(user_list))
            & (self.test_df["asin"].isin(item_list))
        ]

    def prepare_data(self):
        """
        Prepare the training and testing data for the Surprise library by creating Dataset objects.
        """
        # Define the rating scale
        reader = Reader(rating_scale=(1, 5))
        # Load data into Surprise's Dataset object
        self.data = Dataset.load_from_df(
            self.train_df[["user_id", "asin", "rating"]], reader
        )
        # Build the full trainset
        self.trainset = self.data.build_full_trainset()
        # Build the testset
        self.testset = list(
            self.test_df[["user_id", "asin", "rating"]].itertuples(index=False, name=None)
        )

    def train_and_evaluate(self):
        """
        Train and evaluate multiple models with hyperparameter tuning.
        Supported models: SVD, SVD++, NMF, k-NN, Centered k-NN, k-NN Baseline.
        """
        algorithms = {
            "SVD": {
                "algo": SVD,
                "param_grid": {
                    "n_factors": [50, 100],
                    "n_epochs": [20, 30],
                    "lr_all": [0.005, 0.010],
                    "reg_all": [0.02, 0.05],
                },
            },
            "SVD++": {
                "algo": SVDpp,
                "param_grid": {
                    "n_factors": [20, 50],
                    "n_epochs": [10, 20],
                    "lr_all": [0.005],
                    "reg_all": [0.02],
                },
            },
            "NMF": {
                "algo": NMF,
                "param_grid": {
                    "n_factors": [15, 30],
                    "n_epochs": [50, 70],
                    "biased": [False, True],
                },
            },
            "k-NN": {
                "algo": KNNBasic,
                "param_grid": {
                    "k": [20, 40],
                    "min_k": [1, 5],
                    "sim_options": {
                        "name": ["cosine"],
                        "user_based": [False],
                    },
                },
            },
            "Centered k-NN": {
                "algo": KNNWithMeans,
                "param_grid": {
                    "k": [20, 40],
                    "min_k": [1, 5],
                    "sim_options": {
                        "name": ["pearson_baseline"],
                        "user_based": [False],
                    },
                },
            },
            "k-NN Baseline": {
                "algo": KNNBaseline,
                "param_grid": {
                    "k": [20, 40],
                    "min_k": [1, 5],
                    "sim_options": {
                        "name": ["pearson_baseline"],
                        "user_based": [False],
                    },
                },
            },
        }

        # Iterate over each algorithm
        for name, algo_info in algorithms.items():
            print(f"Training {name}...")
            gs = GridSearchCV(
                algo_info["algo"],
                algo_info["param_grid"],
                measures=["rmse", "mae"],
                cv=3,
                n_jobs=-1,
            )
            gs.fit(self.data)

            # Store the best RMSE and MAE
            self.results[name] = {
                "RMSE": gs.best_score["rmse"],
                "MAE": gs.best_score["mae"],
            }

            # Store the best parameters
            self.best_params[name] = gs.best_params["rmse"]

            # Train the algorithm using the best parameters on the full training set
            algo = algo_info["algo"](**gs.best_params["rmse"])
            algo.fit(self.trainset)

            # Evaluate on the test set
            predictions = algo.test(self.testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)

            # Update results with test set performance
            self.results[name]["Test_RMSE"] = rmse
            self.results[name]["Test_MAE"] = mae

        # Print the results
        for name, metrics in self.results.items():
            print(f"{name}:")
            print(f"  Best RMSE (CV): {metrics['RMSE']:.4f}")
            print(f"  Best MAE (CV): {metrics['MAE']:.4f}")
            print(f"  Test RMSE: {metrics['Test_RMSE']:.4f}")
            print(f"  Test MAE: {metrics['Test_MAE']:.4f}")
            print(f"  Best Params: {self.best_params[name]}\n")

    def plot_results(self):
        """
        Plot the performance of different models on the test set.
        """
        # Prepare data for plotting
        res = []
        for model_name, metrics in self.results.items():
            res.append(
                {
                    "model": model_name,
                    "RMSE": metrics["Test_RMSE"],
                    "MAE": metrics["Test_MAE"],
                }
            )
        res_df = pd.DataFrame(res)

        # Reshape the data for plotting
        res_long = res_df.melt(
            id_vars="model", value_vars=["RMSE", "MAE"], var_name="metric"
        )

        # Plot the performance using Seaborn
        plt.figure(figsize=(12, 6))
        sns.barplot(x="model", y="value", hue="metric", data=res_long)
        plt.xlabel("Model")
        plt.ylabel("Value")
        plt.title("Performance of Different Collaborative Filtering Models on Test Dataset")
        plt.legend(title="Metric")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Execute the entire workflow: data loading, preprocessing, model training,
        hyperparameter tuning, evaluation, and result visualization.
        """
        self.load_data()
        self.split_data()
        self.filter_data()
        self.prepare_data()
        self.train_and_evaluate()
        self.plot_results()


# Usage example
if __name__ == "__main__":
    # Initialize the trainer with the path to your data file
    trainer = SurpriseModelTrainer("Beauty_and_Personal_Care_2022.parquet")
    # Run the training and evaluation process
    trainer.run()
