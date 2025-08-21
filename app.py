import os
import matplotlib
matplotlib.use('Agg')  # forza backend senza GUI su macOS
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier, plot_tree

app = Flask(__name__)

# Dataset di esempio: parole, link, emoji
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

# Funzione per estrarre feature dal testo
def extract_features(text):
    words = len(text.split())
    links = text.count("http")
    emojis = sum(1 for c in text if c in "😀😂😎😍👍")
    return [words, links, emojis]

# Genera immagine dell'albero senza aprire finestre
def plot_tree_image():
    plt.figure(figsize=(6,4))
    plot_tree(clf, feature_names=["words","links","emojis"], class_names=clf.classes_, filled=True)
    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig("static/tree.png")
    plt.close()

# Rotta principale
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
    app.run(debug=True, host="127.0.0.1", port=5000)
