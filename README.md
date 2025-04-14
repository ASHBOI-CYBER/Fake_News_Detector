# Fake News Detection Web App

This project uses a Random Forest Classifier to detect whether a news article is real or fake based on its title and content. The model is deployed as a simple web app where users can input a news article and get a prediction.

## Features

- Data preprocessing using TF-IDF vectorization
- Training with a Random Forest Classifier
- Cross-validation for performance evaluation
- Simple web interface (e.g., Flask or Streamlit)
- Model and vectorizer saved as `.pkl` files for reuse

## Project Structure

    ├── rf_model.pkl               # Trained Random Forest model
    ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer used for preprocessing
    ├── requirements.txt           # Python package dependencies
    ├── app.py / notebook.ipynb    # Main Python or Jupyter notebook file
    └── README.md                  # Project description

## Dataset

You can get the dataset from:  
[Kaggle - Fake News Detection](https://www.kaggle.com/competitions/fake-news/data)

Make sure the dataset includes the following columns:
- title: Title of the news article
- text: Full article text
- label: 0 for real, 1 for fake

## How to Run

1. Install requirements:

       pip install -r requirements.txt

2. Train the model (if not already trained):
       with open("rf_model.pkl", "wb") as f:
           pickle.dump(model, f)
       with open("tfidf_vectorizer.pkl", "wb") as f:
           pickle.dump(vectorizer, f)

3. Run the web app (example using Streamlit):
       streamlit run app.py

## Accuracy

The model achieved:
- Test Accuracy: ~99.71%
- Cross-Validation Mean Accuracy: ~99.47%

## Note

Ensure `rf_model.pkl` and `tfidf_vectorizer.pkl` are in the same directory as your app when deploying.

## License

This project is for educational purposes. You are free to modify and expand it as needed.
