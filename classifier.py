from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load the iris dataset from sklearn"""
    iris = load_iris()
    return iris.data, iris.target, iris.target_names

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing datasets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """Train a Logistic regression classifier"""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate the model's performance on test data"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"\n Accuracy: {accuracy:4f}")
    print("\n Classification report:\n", report)

def main():
    """Main function to run the classification pipeline"""
    print("Loading iris dataset...")
    X, y, target_names = load_data()

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, target_names)

if __name__ == "__main__":
    main()