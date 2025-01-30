import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv(url, header=None, names=columns)

# Step 2: Basic Exploration
print("\nDataset Preview:")
print(iris.head())
print("\nDataset Info:")
print(iris.info())
print("\nSummary Statistics:")
print(iris.describe())

# Step 3: Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Convert data types if necessary (none required for this dataset)
# iris['column_name'] = iris['column_name'].astype(desired_type)

# Step 4: Descriptive Statistics
mean_values = iris.mean(numeric_only=True)
median_values = iris.median(numeric_only=True)
std_values = iris.std(numeric_only=True)

print("\nMean Values:\n", mean_values)
print("\nMedian Values:\n", median_values)
print("\nStandard Deviation Values:\n", std_values)

# Step 5: Data Visualization
# Histogram
iris.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Iris Dataset Features", y=0.95, fontsize=16)
plt.show()

# Scatter Plot
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
plt.suptitle("Scatter Plot Matrix", y=1.02, fontsize=16)
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris.drop(columns=["species"]))
plt.title("Box Plots of Features", fontsize=16)
plt.show()

# Optional: EDA
# Correlation Matrix
correlation_matrix = iris.drop(columns="species").corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# Identify outliers using box plots (already visualized above)
# Initial observations about the dataset
print("\nInitial Observations:")
print("- Sepal length and petal length show strong positive correlation.")
print("- Species appears to cluster distinctly in feature space.")
# End of script for data exploration, visualization, and optional EDA.

# Additional Step: Notes Taking API (Production Level Implementation Placeholder)
# The following is a placeholder for implementing a notes-taking API
from flask import Flask, request, jsonify

app = Flask(__name__)

notes = []

@app.route('/notes', methods=['GET', 'POST'])
def manage_notes():
    if request.method == 'POST':
        note = request.json.get('note')
        if note:
            notes.append(note)
            return jsonify({"message": "Note added successfully.", "notes": notes}), 201
        return jsonify({"error": "No note provided."}), 400
    return jsonify({"notes": notes})

@app.route('/notes/<int:note_id>', methods=['GET', 'DELETE'])
def note_detail(note_id):
    if 0 <= note_id < len(notes):
        if request.method == 'DELETE':
            removed_note = notes.pop(note_id)
            return jsonify({"message": "Note deleted successfully.", "removed_note": removed_note}), 200
        return jsonify({"note": notes[note_id]}), 200
    return jsonify({"error": "Note not found."}), 404

if __name__ == '__main__':
    app.run(debug=True)