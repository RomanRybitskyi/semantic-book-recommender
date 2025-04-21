# Book Recommender System

This project implements a semantic book recommender system that suggests books based on user queries, categories, and emotional tones. It leverages natural language processing (NLP) and vector search to provide personalized book recommendations. The system processes a dataset of approximately 7,000 books, performs data cleaning, text classification, sentiment analysis, and semantic search, and presents results through an interactive Gradio dashboard.

## Project Structure

The project consists of several Python scripts and Jupyter notebooks, each handling a specific part of the pipeline:

- **`data-exploration.ipynb`**: Loads and preprocesses the book dataset, performs exploratory data analysis, and saves a cleaned dataset (`book_cleaned.csv`).
- **`text-classification.ipynb`**: Classifies books into simplified categories (e.g., Fiction, Nonfiction) using a zero-shot classification model and saves the results (`books_with_categories.csv`).
- **`sentiment-analysis.ipynb`**: Analyzes the emotional tone of book descriptions using a text classification model and saves the results (`books_with_emotions.csv`).
- **`vector-search.ipynb`**: Implements semantic search using LangChain and Chroma to retrieve book recommendations based on user queries.
- **`gradio-dashboard.py`**: Creates an interactive web interface using Gradio to allow users to input queries, select categories and tones, and view recommendations.
- **`.env`**: Stores the Hugging Face API token for accessing models.

## Features

- **Semantic Search**: Uses sentence embeddings to match user queries with book descriptions.
- **Category Filtering**: Allows users to filter recommendations by categories like Fiction, Nonfiction, Children's Fiction, etc.
- **Emotional Tone Sorting**: Sorts recommendations based on emotional tones (e.g., Happy, Sad, Suspenseful) derived from sentiment analysis.
- **Interactive Dashboard**: Provides a user-friendly Gradio interface to input queries and visualize recommendations with book covers and descriptions.

## Dataset

The dataset is sourced from Kaggle (`dylanjcastillo/7k-books-with-metadata`) and includes metadata for approximately 7,000 books, such as:
- ISBN13, title, subtitle, authors, description, categories, published year, average rating, number of pages, and thumbnail.
- The dataset is cleaned to remove missing values, filter descriptions with at least 25 words, and create new features like `age_of_book` and `title_and_subtitle`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RomanRybitskyi/semantic-book-recommender.git
   cd book-recommender-system
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the following key libraries:
   - `pandas`, `numpy`, `seaborn`, `matplotlib` for data processing and visualization.
   - `transformers`, `langchain`, `chroma`, `huggingface_hub` for NLP and vector search.
   - `gradio` for the dashboard.
   - `kagglehub` for dataset download.
   - `torch` with CUDA support for GPU acceleration (optional).

4. **Set up environment variables**:
   Create a `.env` file in the project root and add your Hugging Face API token:
   ```
   HUGGINGFACE_TOKEN=<your-huggingface-token>
   ```

5. **Download the dataset**:
   Run `data-exploration.ipynb` to download the dataset using `kagglehub`. Ensure you have a Kaggle account and API token configured if required.

## Usage

1. **Run the preprocessing notebooks**:
   - Execute `data-exploration.ipynb` to clean the dataset and generate `book_cleaned.csv`.
   - Run `text-classification.ipynb` to classify categories and generate `books_with_categories.csv`.
   - Run `sentiment-analysis.ipynb` to analyze emotions and generate `books_with_emotions.csv`.

2. **Test semantic search**:
   - Run `vector-search.ipynb` to test semantic search functionality with sample queries (e.g., "A book about WW2").

3. **Launch the Gradio dashboard**:
   ```bash
   python gradio-dashboard.py
   ```
   - Open the provided URL in your browser.
   - Enter a query (e.g., "A story about forgiveness"), select a category and tone, and click "Find recommendations" to view results.

## Example

**Query**: "A book about WW2"  
**Category**: Nonfiction  
**Tone**: Sad  

The system retrieves up to 16 books with descriptions semantically similar to the query, filtered by the Nonfiction category, and sorted by the "sadness" score of their descriptions. Results are displayed with book covers, titles, authors, and truncated descriptions.

## Requirements

See `requirements.txt` for a complete list of dependencies. Key libraries include:
- Python 3.8+
- `pandas>=2.0.0`
- `transformers>=4.30.0`
- `langchain>=0.0.200`
- `chroma>=0.4.0`
- `gradio>=3.30.0`
- `kagglehub>=0.1.0`

## Limitations

- **Dataset Quality**: The system relies on the quality and completeness of book descriptions. Missing or short descriptions may affect recommendation accuracy.
- **Category Mapping**: Only a subset of categories is mapped to simplified categories, which may exclude some niche genres.
- **Computational Resources**: Sentiment analysis and zero-shot classification are computationally intensive and benefit from GPU acceleration.
- **Tone Sorting**: Emotional tone analysis is based on maximum emotion scores per description, which may not capture nuanced sentiments.

## Future Improvements

- Expand category mappings to include more genres (e.g., Romance, Fantasy).
- Incorporate user ratings or popularity metrics to refine recommendations.
- Optimize semantic search by experimenting with different embedding models.
- Add support for multiple languages or multilingual embeddings.
- Enhance the Gradio dashboard with advanced filters (e.g., publication year, page count).
- Add more books to dataset

## Acknowledgments

- Dataset provided by [Dylan J. Castillo](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata).
- Powered by Hugging Face Transformers, LangChain, Chroma, and Gradio.