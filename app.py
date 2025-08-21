import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier, plot_tree
import emoji

app = Flask(__name__)

# dataset più grande
X = [
    [20, 0, 0],   # breve, no link, no emoji → da leggere
    [40, 0, 0],   # breve medio → da leggere
    [80, 0, 0],   # medio → da leggere
    [150, 1, 0],  # un po' lungo con link → urgente
    [200, 2, 0],  # lungo con più link → urgente
    [250, 3, 1],  # molto lungo con link ed emoji → urgente
    [50, 0, 3],   # breve con emoji → spam
    [100, 0, 4],  # medio con emoji → spam
    [120, 1, 5],  # medio con link + emoji → spam
    [300, 0, 6],  # molto lungo con emoji → spam
    [400, 0, 8],  # lunghissimo con emoji → spam
    [350, 2, 0],  # molto lungo con link → urgente
    [15, 0, 1],   # cortissimo con emoji → spam
    [60, 1, 0],   # breve con link → urgente
    [90, 0, 2],   # medio con emoji → spam
]

y = [
    "da leggere",
    "da leggere",
    "da leggere",
    "urgente",
    "urgente",
    "urgente",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "urgente",
    "spam",
    "urgente",
    "spam",
]

clf = DecisionTreeClassifier()
clf.fit(X, y)

def extract_features(text):
    words = len(text.split())
    links = text.count("http")
    emojis = sum(1 for c in text if c in emoji.EMOJI_DATA)
    return [words, links, emojis]

def plot_tree_image():
    plt.figure(figsize=(12,9), dpi=100)
    plot_tree(
        clf,
        feature_names=["words", "links", "emojis"],
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        fontsize=12
    )
    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig("static/tree.png")
    plt.close()
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        text = request.form["email_text"]
        features = extract_features(text)
        prediction = clf.predict([features])[0]
        plot_tree_image()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
