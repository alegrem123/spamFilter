from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Dataset esempio: parole, link, emoji
X = [
    [50, 0, 0],
    [200, 2, 1],
    [500, 0, 3],
    [30, 1, 0],
    [400, 3, 5]
]

y = ["da leggere", "urgente", "spam", "da leggere", "spam"]

# Allena il modello
clf = DecisionTreeClassifier()
clf.fit(X, y)

def extract_features(text):
    words = len(text.split())
    links = text.count("http")
    emojis = sum(1 for c in text if c in "😀😂😎😍👍")
    return [words, links, emojis]

def plot_tree_image():
    plt.figure(figsize=(6,4))
    plot_tree(clf, feature_names=["words","links","emojis"], class_names=clf.classes_, filled=True)
    plt.savefig("static/tree.png")
    plt.close()

@app.route("/", methods=["GET","POST"])
def index():
    prediction = ""
    if request.method == "POST":
        text = request.form["email_text"]
        features = extract_features(text)
        prediction = clf.predict([features])[0]
        plot_tree_image()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
