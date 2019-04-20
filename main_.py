# Importing libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Handling exceptions
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Loading dataset
data = pd.read_csv("data.csv")
data.head()

# Featuring & labelling
y = data["label"]
url_list = data["url"]

# Tokenizing and fitting the vectors into variable X
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(url_list)

# Spliting data into 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)

# Evaluating model's accuracy
print("Accuracy of model is: ",logit.score(X_test, y_test))

