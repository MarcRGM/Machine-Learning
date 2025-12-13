import pandas as pd # Pandas reads CSV files into tables
from sklearn.model_selection import train_test_split
# Gets the train_test_split function that splits data into train (80%) + test (20%)
from sklearn.ensemble import RandomForestClassifier
# Gets the RandomForestClassifier class taht learns hand pose
import joblib # Library for saving/loading trained models

hand_dataset = pd.read_csv("data/sign_data.csv") 
# Reads the CSV into a table called hand_dataset
X = hand_dataset.iloc[:, :-1]  # 63 landmark values
# all rows (:), all columns except last (:-1)
y = hand_dataset.iloc[:, -1]   # labels
# all rows, only last column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Split data: 80% to train (learn patterns), 20% to test (check accuracy on unseen data)
model = RandomForestClassifier() # Creates a new, empty RandomForest model
model.fit(X_train, y_train) # Trains the model

print(f"Accuracy: {model.score(X_test, y_test):.2f}") 
# % correct predictions on unseen test data, .2f = show 2 decimal places
joblib.dump(model, "model.pkl")  # Save for later use
# Saves trained model to model.pkl file