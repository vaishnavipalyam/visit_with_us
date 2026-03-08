# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/VaishnaviPalyam/visit_with_us/data/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Correct gender column
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',               # Customer's age
    'NumberOfPersonVisiting',           # Total number of people accompanying the customer on the trip
    'NumberOfTrips',     # Average number of trips the customer takes annually
    'NumberOfChildrenVisiting',         # children below age 5 accompanying the customer
    'MonthlyIncome',    # Customer’s estimated salary per month
    'DurationOfPitch', # Duration of the sales pitch delivered
    'CityTier', #Tiering of the residential city
    'PreferredPropertyStar', #Preferred property
    'Passport', # Holds passport or not (binary: 0:does not have passport, 1:Holds passport)
    'PitchSatisfactionScore', #Pitch satisfaction feedback from customer
    'OwnCar', # If the customer owns a car (binary, 0: does not own one, 1: owns a car)
    'NumberOfFollowups' # Total number of follow ups by the sales person
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer contacted
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="data/"+file_path.split("/")[-1],
        repo_id="VaishnaviPalyam/visit_with_us",
        repo_type="dataset",
    )
