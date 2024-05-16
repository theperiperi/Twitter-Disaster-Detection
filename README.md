# Twitter Disaster Detection

## Overview
This project aims to detect tweets that indicate a disaster using machine learning techniques. It utilizes a dataset containing tweets labeled as either indicating a disaster or not, and trains a classifier to predict whether new tweets from a test dataset indicate a disaster.

## Dataset
- `train.csv`: Contains labeled tweets used for training the model.
- `test.csv`: Contains tweets to predict whether they indicate a disaster.

## Dependencies
- Python 3.x
- pandas
- scikit-learn

## Installation
1. Clone this repository:
git clone https://github.com/theperiperi/Twitter-Disaster-Detection.git
2. Navigate to the project directory:
cd twitter-disaster-detection
3. Install dependencies:
pip install -r requirements.txt


## Usage
1. Place the `train.csv` and `test.csv` files in the project directory.
2. Run the following command to train the model and make predictions:
python detect_disasters.py

3. The predictions will be saved to `predictions.csv`.

## Model
- The model used for this project is a logistic regression classifier.
- Text data is preprocessed and converted into numerical features using TF-IDF vectorization.
- The classifier is trained on the labeled tweets from the `train.csv` file.

## Results
- The model's performance can be evaluated using accuracy metrics.
- Training accuracy is calculated by comparing the model's predictions on the training data to the actual labels.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
