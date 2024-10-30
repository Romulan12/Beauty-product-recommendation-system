# Beauty Recommendation System

The Beauty Recommendation System utilizes large-scale datasets containing beauty product information, user reviews, and purchase histories to build a recommendation engine. The system incorporates collaborative filtering and vector-based similarity search to suggest products that align with user preferences.

## Directory Structure

```
├── README.md
├── code
│   ├── app.py
│   ├── collaborative_models.py
│   └── vector_db.py
├── data
│   ├── All_Beauty.parquet
│   ├── Beauty_and_Personal_Care.parquet
│   ├── ...
│   └── train_data.parquet
├── notebooks
│   ├── EDA.ipynb
│   ├── cleanup_data.ipynb
│   ├── get_vector_db_performance.ipynb
│   └── parse_jsonl.ipynb
```

- **`code/`**: Contains the source code for the application and model implementations.
  - **`app.py`**: The main application script to run the recommendation system.
  - **`collaborative_models.py`**: Implements collaborative filtering algorithms.
  - **`vector_db.py`**: Handles vector database operations for similarity search.
- **`data/`**: Includes all the datasets used for training and evaluation.
- **`notebooks/`**: Jupyter notebooks for data exploration, cleaning, and analysis.
  - **`EDA.ipynb`**: Exploratory Data Analysis of the datasets.
  - **`cleanup_data.ipynb`**: Data cleaning and preprocessing steps.
  - **`get_vector_db_performance.ipynb`**: Performance evaluation of vector databases.
  - **`parse_jsonl.ipynb`**: Scripts to parse raw JSONL data files.

## Getting Started

### Prerequisites

- Python 3.12
- Required Python packages listed in `requirements.txt`

>   Place the required data files into the `data/` directory. Due to their size, they are not included in the repository. [Link for Data](https://drive.google.com/drive/folders/1xWNs_38NOzqSuzRlF_dNShDadRnm96bw?usp=sharing)

### Running the Application

Execute the main application script:

```bash
streamlit run code/app.py
```

This will start the recommendation system. Follow the on-screen instructions or access the web interface if available.

## Data Description

The `data/` directory contains:

- **Product Data:**
  - `All_Beauty.parquet`
  - `Beauty_and_Personal_Care.parquet`
  - `meta_beauty.parquet`
  - Includes product details, categories, and metadata.

- **User Data:**
  - `purchase_history.parquet`
  - Contains anonymized user purchase histories.

- **Processed Data:**
  - `combined_with_embeddings.parquet`
  - Datasets enriched with vector embeddings for similarity computations.

 >  ChatGPT was used in this project to assist with debugging Python code, generating docstring explanations, and reviewing grammar in the report. All AI outputs were carefully reviewed, modified as necessary, and responsibly integrated to maintain academic integrity.
