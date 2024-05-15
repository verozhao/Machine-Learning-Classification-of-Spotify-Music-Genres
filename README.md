# Machine Learning Classification of Spotify Music Genres

## Introduction
This project tackles the task of classifying music genres using Spotify's dataset of diverse audio features, employing various machine learning models to effectively predict musical genres.

## EDA and Data Preprocessing
Extensive EDA and preprocessing were conducted to ensure the dataset was optimally prepared for ML models, addressing various challenges critical for maintaining the integrity and usability of the dataset.

### Correlation Analysis
Two correlation matrices were plotted to understand relationships between variables, which guided the selection of features for the models. We observed that feature `artist_name` was strongly correlated with `music_genre`.

### Feature Engineering and Transformation
Five completely empty rows were identified and removed to enhance data quality. Notable missing values in the `duration_ms` attribute, marked as `-1`, were replaced with the median value derived from the training data to ensure consistency. Similarly, missing entries in the `tempo` feature, indicated by `"?"`, were appropriately addressed. Categorical variables such as `key` and `mode` underwent one-hot encoding to convert them into numerical formats suitable for machine learning algorithms, while the target variable, `music_genre`, was transformed from categorical labels into numeric labels through label encoding. These transformations were crucial for enabling the application of various machine learning techniques that require numerical input.

### Preventing Data Leakage
To ensure the integrity of the model evaluation, the data was divided into training and testing sets before any preprocessing tasks were conducted. This division was carefully managed to guarantee balanced representation across genres, with each genre contributing 500 songs to the test set and 4,500 to the training set.

### Dimensionality Reduction
Due to high dimensionality from one-hot encoding, Principal Component Analysis (PCA) was used to reduce dimensions while retaining 95% of variance, enhancing computational efficiency and revealing underlying data structures.

## Model Selection and Implementation
Multiple models were implemented, including SVM, Decision Trees, Random Forest, Gradient Boosting, and Neural Networks. The Neural Network was designed with three layers using ReLU and softmax activations and was implemented in Keras, while the other models were developed using Scikit-Learn. Each model was carefully selected for its unique capabilities in handling the complexities of multi-class classification and high-dimensional data.

## Model Evaluation and Analysis

### AUC ROC
Models were evaluated with SVM and Neural Networks being particularly effective, showcasing superior performance in high-dimensional settings indicated by their AUC scores. The final AUC scores were:

| Model              | Micro-Average AUC | Macro-Average AUC |
|--------------------|-------------------|-------------------|
| Neural Network     | 0.94              | 0.94              |
| SVM                | 0.97              | 0.96              |
| Decision Tree      | 0.69              | 0.69              |
| Random Forest      | 0.93              | 0.92              |
| Gradient Boosting  | 0.94              | 0.93              |

### Analysis of Confusion Matrix and Classification Reports
The model evaluations detailed their precision, recall, and F1-scores across ten music genres, providing key insights:

- **Neural Network:** Demonstrated a balanced 71% accuracy, excelling in Classical (F1-score of 0.90) and Jazz but faltering in genres like Metal and Hip-Hop.
- **SVM:** Emerged as the top performer with a 73% accuracy, showing high precision and recall, especially in Pop and Electronic music.
- **Decision Trees:** While only achieving 44% accuracy, offered insights into feature importance but showed signs of overfitting.
- **Random Forest:** Improved upon Decision Trees with a 59% accuracy, providing better generalization across genres.
- **Gradient Boosting:** Achieved a solid 61% accuracy, effectively handling genres with subtle distinctions through iterative error correction.

### Feature Importance Analysis
The evaluation of feature importance across different models highlighted `artist_name` as a significant predictor, particularly in Decision Trees, Random Forest, and Gradient Boosting. This finding suggests that the identity of the artist is a strong indicator of the genre, likely due to artists typically sticking to specific genre styles.

## Conclusion
The project successfully applies advanced machine learning techniques to music genre classification, highlighting the impact of dimensionality reduction, meticulous data handling, and strategic model selection. Robust AUC scores reflect these efforts, with the primary success factor being thorough feature engineering and selection that enhanced model performance.
