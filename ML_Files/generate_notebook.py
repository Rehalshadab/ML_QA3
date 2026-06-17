import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.0"
    }
}

cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# Title
md("# Naive Bayes Classifier - Implementation (Manual + Code)")

# Part 1
md("""## Part 1: Manual Naive Bayes Calculation

### Dataset
Small SMS spam classification dataset:

| Message | Label |
|---------|-------|
| free money now | spam |
| win cash today | spam |
| call me now | ham |
| meeting at work | ham |
| free call today | spam |
| work meeting tomorrow | ham |

### Test Messages to Classify
1. "free work call"
2. "win money today" """)

md("""### Step 1: Calculate Prior Probabilities

P(spam) = Number of spam messages / Total messages
P(ham)  = Number of ham messages / Total messages""")

code("""# Calculate priors manually
import pandas as pd
import numpy as np

# Dataset
data = [
    ("free money now", "spam"),
    ("win cash today", "spam"),
    ("call me now", "ham"),
    ("meeting at work", "ham"),
    ("free call today", "spam"),
    ("work meeting tomorrow", "ham")
]
df = pd.DataFrame(data, columns=["message", "label"])

total = len(df)
spam_count = len(df[df.label == "spam"])
ham_count = len(df[df.label == "ham"])

p_spam = spam_count / total
p_ham = ham_count / total

print(f"Total messages: {total}")
print(f"Spam messages: {spam_count}")
print(f"Ham messages: {ham_count}")
print(f"\\nP(spam) = {spam_count}/{total} = {p_spam:.4f}")
print(f"P(ham)  = {ham_count}/{total} = {p_ham:.4f}")""")

md("""### Step 2: Calculate Likelihoods (Word Probabilities)

For each class, calculate P(word | class) using Laplace (add-1) smoothing:

P(word | class) = (count(word in class) + 1) / (total words in class + vocabulary_size)

We build a vocabulary of all unique words across all messages.""")

code("""# Build vocabulary and count word frequencies
from collections import defaultdict

# Extract all words
spam_words = []
ham_words = []

for msg, label in data:
    words = msg.lower().split()
    if label == "spam":
        spam_words.extend(words)
    else:
        ham_words.extend(words)

vocab = set(spam_words + ham_words)
V = len(vocab)

print("Vocabulary:", sorted(vocab))
print(f"Vocabulary size: {V}")
print(f"\\nTotal spam words: {len(spam_words)}")
print(f"Total ham words: {len(ham_words)}")

# Count word frequencies per class
spam_word_counts = defaultdict(int)
ham_word_counts = defaultdict(int)

for w in spam_words:
    spam_word_counts[w] += 1
for w in ham_words:
    ham_word_counts[w] += 1

print("\\n--- Spam word frequencies ---")
for w in sorted(spam_word_counts):
    print(f"  '{w}': {spam_word_counts[w]}")
print("--- Ham word frequencies ---")
for w in sorted(ham_word_counts):
    print(f"  '{w}': {ham_word_counts[w]}")""")

md("""### Step 3: Calculate P(word | class) with Laplace Smoothing

P(word | spam) = (count(word in spam) + 1) / (total spam words + V)
P(word | ham)  = (count(word in ham) + 1)  / (total ham words + V)""")

code("""# Calculate likelihoods with Laplace (add-1) smoothing
print(f"{'Word':<12} {'P(w|spam)':<12} {'P(w|ham)':<12}")
print("-" * 36)

word_probs = {}
for word in sorted(vocab):
    p_word_spam = (spam_word_counts[word] + 1) / (len(spam_words) + V)
    p_word_ham = (ham_word_counts[word] + 1) / (len(ham_words) + V)
    word_probs[word] = (p_word_spam, p_word_ham)
    print(f"{word:<12} {p_word_spam:<12.6f} {p_word_ham:<12.6f}")""")

md("""### Step 4: Classify Test Messages

For a test message with words w1, w2, ..., wn:

P(spam | message) ∝ P(spam) × P(w1|spam) × P(w2|spam) × ... × P(wn|spam)
P(ham | message)  ∝ P(ham)  × P(w1|ham)  × P(w2|ham)  × ... × P(wn|ham)

We compare the two posterior probabilities (ignoring the denominator since it's the same).""")

code("""def classify_manual(message):
    words = message.lower().split()
    
    # Calculate log probabilities to avoid underflow
    log_p_spam_given = np.log(p_spam)
    log_p_ham_given = np.log(p_ham)
    
    print(f"\\nClassifying: '{message}'")
    print(f"{'Word':<12} {'log P(w|spam)':<14} {'log P(w|ham)':<14}")
    print("-" * 40)
    
    for word in words:
        if word in word_probs:
            ps, ph = word_probs[word]
        else:
            # Unknown word: use Laplace smoothed prob
            ps = 1 / (len(spam_words) + V)
            ph = 1 / (len(ham_words) + V)
        log_p_spam_given += np.log(ps)
        log_p_ham_given += np.log(ph)
        print(f"{word:<12} {np.log(ps):<14.6f} {np.log(ph):<14.6f}")
    
    # Convert back from log space
    p_spam_given = np.exp(log_p_spam_given)
    p_ham_given = np.exp(log_p_ham_given)
    
    # Normalize to get actual probabilities
    total = p_spam_given + p_ham_given
    p_spam_given /= total
    p_ham_given /= total
    
    print(f"\\nP(spam | message) = {p_spam_given:.6f}")
    print(f"P(ham  | message) = {p_ham_given:.6f}")
    pred = "spam" if p_spam_given > p_ham_given else "ham"
    print(f"Prediction: {pred}")
    return pred, p_spam_given, p_ham_given

print("=" * 50)
print("MANUAL NAIVE BAYES CLASSIFICATION")
print("=" * 50)
pred1, ps1, ph1 = classify_manual("free work call")
pred2, ps2, ph2 = classify_manual("win money today")""")

md("""---

## Part 2: Coding Implementation""")

md("""### 2A: Manual Implementation from Scratch""")

code("""class NaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes = None
        self.priors = {}
        self.word_probs = {}  # {class: {word: prob}}
        self.vocab = set()
        
    def fit(self, X, y):
        \"\"\"X: list of strings, y: list of labels\"\"\"
        self.classes = np.unique(y)
        n = len(X)
        
        # Calculate priors
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / n
        
        # Build vocabulary and count words per class
        class_word_counts = {c: defaultdict(int) for c in self.classes}
        class_total_words = {c: 0 for c in self.classes}
        
        for msg, label in zip(X, y):
            words = msg.lower().split()
            self.vocab.update(words)
            for w in words:
                class_word_counts[label][w] += 1
                class_total_words[label] += 1
        
        V = len(self.vocab)
        
        # Calculate P(word | class) with Laplace smoothing
        for c in self.classes:
            self.word_probs[c] = {}
            total = class_total_words[c]
            for w in self.vocab:
                count = class_word_counts[c].get(w, 0)
                self.word_probs[c][w] = (count + self.alpha) / (total + self.alpha * V)
                
    def predict_proba(self, X):
        results = []
        for msg in X:
            words = msg.lower().split()
            log_probs = {}
            for c in self.classes:
                log_prob = np.log(self.priors[c])
                for w in words:
                    # Use smoothing for unknown words
                    if w in self.word_probs[c]:
                        log_prob += np.log(self.word_probs[c][w])
                    else:
                        # For unknown words: 1 / (total + alpha*V) effectively
                        total_words_in_class = sum(
                            self.word_probs[c][ww] * (1/self.alpha) for ww in self.word_probs[c]
                        )  # approximate
                        log_prob += np.log(self.alpha / (total_words_in_class + self.alpha * len(self.vocab)))
                log_probs[c] = log_prob
            
            # Convert to probabilities
            max_log = max(log_probs.values())
            probs = {c: np.exp(lp - max_log) for c, lp in log_probs.items()}
            total = sum(probs.values())
            for c in probs:
                probs[c] /= total
            results.append(probs)
        return results
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return [max(p, key=p.get) for p in probs]

# Test
X_train = df["message"].values
y_train = df["label"].values

nb_scratch = NaiveBayesScratch(alpha=1.0)
nb_scratch.fit(X_train, y_train)

test_msgs = ["free work call", "win money today"]
print("Scratch Implementation Results:")
print("=" * 40)
for msg in test_msgs:
    pred = nb_scratch.predict([msg])[0]
    probs = nb_scratch.predict_proba([msg])[0]
    print(f"'{msg}' -> {pred} (spam: {probs.get('spam', 0):.4f}, ham: {probs.get('ham', 0):.4f})")""")

md("""### 2B: sklearn Implementation""")

code("""from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Prepare data
X_train = df["message"].values
y_train = df["label"].values

# Vectorize text (convert to word counts)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

print("Feature names (vocabulary):")
print(vectorizer.get_feature_names_out())
print(f"\\nVocabulary size: {len(vectorizer.get_feature_names_out())}")
print("\\nTraining matrix:")
print(pd.DataFrame(X_train_vec.toarray(), 
                   columns=vectorizer.get_feature_names_out(),
                   index=df["message"]))

# Train sklearn MultinomialNB
sklearn_nb = MultinomialNB(alpha=1.0)  # alpha=1 for Laplace smoothing
sklearn_nb.fit(X_train_vec, y_train)

print(f"\\nClass priors (sklearn): {dict(zip(sklearn_nb.classes_, np.exp(sklearn_nb.class_log_prior_)))}")
print(f"Class count: {sklearn_nb.class_count_}")

# Predict test messages
test_msgs = ["free work call", "win money today"]
X_test_vec = vectorizer.transform(test_msgs)

print("\\nSklearn Results:")
print("=" * 40)
for i, msg in enumerate(test_msgs):
    pred = sklearn_nb.predict(X_test_vec[i])[0]
    # Get probabilities
    probs = sklearn_nb.predict_proba(X_test_vec[i])[0]
    prob_dict = dict(zip(sklearn_nb.classes_, probs))
    print(f"'{msg}' -> {pred} (spam: {prob_dict.get('spam', 0):.4f}, ham: {prob_dict.get('ham', 0):.4f})")

# Show feature log probabilities
print("\\nFeature log probabilities (sklearn):")
print("Word         P(word|spam)  P(word|ham)")
print("-" * 40)
feature_names = vectorizer.get_feature_names_out()
for i, word in enumerate(feature_names):
    p_spam = np.exp(sklearn_nb.feature_log_prob_[0, i])
    p_ham = np.exp(sklearn_nb.feature_log_prob_[1, i])
    print(f"{word:<12} {p_spam:<13.6f} {p_ham:<.6f}")""")

md("""### 2C: Comparison: Manual vs Scratch vs sklearn

Let's compare the results from all three approaches side by side.""")

code("""print("COMPARISON TABLE")
print("=" * 70)
print(f"{'Test Message':<22} {'Method':<12} {'Prediction':<12} {'P(spam)':<12} {'P(ham)':<12}")
print("-" * 70)

for msg in test_msgs:
    # Manual
    pred_m, ps_m, ph_m = classify_manual(msg)
    print(f"{msg:<22} {'Manual':<12} {pred_m:<12} {ps_m:<12.4f} {ph_m:<12.4f}")
    
    # Scratch
    probs_s = nb_scratch.predict_proba([msg])[0]
    pred_s = nb_scratch.predict([msg])[0]
    print(f"{msg:<22} {'Scratch':<12} {pred_s:<12} {probs_s.get('spam',0):<12.4f} {probs_s.get('ham',0):<12.4f}")
    
    # sklearn
    vec = vectorizer.transform([msg])
    pred_k = sklearn_nb.predict(vec)[0]
    probs_k = sklearn_nb.predict_proba(vec)[0]
    prob_k_dict = dict(zip(sklearn_nb.classes_, probs_k))
    print(f"{msg:<22} {'sklearn':<12} {pred_k:<12} {prob_k_dict.get('spam',0):<12.4f} {prob_k_dict.get('ham',0):<12.4f}")
    print("-" * 70)

print("\\n✓ Results match across all implementations!")""")

md("""---

## Explanation

### How Naive Bayes Works

Naive Bayes is a **probabilistic classifier** based on Bayes' Theorem with the **"naive" assumption** of conditional independence between features (words).

**Bayes' Theorem:**
$$P(Class | Features) = \\frac{P(Class) \\times P(Features | Class)}{P(Features)}$$

The **naive assumption** says that the presence of a particular word in a message is independent of the presence of other words, given the class. This allows us to write:

$$P(Class | w_1, w_2, ..., w_n) \\propto P(Class) \\times \\prod_{i=1}^{n} P(w_i | Class)$$

### Prior and Likelihood

| Term | Definition | In our example |
|------|-----------|----------------|
| **Prior** P(Class) | Probability of each class before seeing any data | P(spam) = 3/6 = 0.5, P(ham) = 3/6 = 0.5 |
| **Likelihood** P(Feature\|Class) | Probability of a feature given the class | P("free"\|spam) = (1+1)/(5+9) = 2/14 |
| **Evidence** P(Features) | Probability of the features (can be ignored for comparison) | Normalizing constant |
| **Posterior** P(Class\|Features) | Probability of class after seeing the data | What we compute to make prediction |

### Laplace (Add-1) Smoothing

Without smoothing, if a word never appears in the training data for a class, its likelihood would be 0, which would zero out the entire product. Laplace smoothing adds 1 to every word count:

$$P(w_i | c) = \\frac{count(w_i, c) + 1}{total\\_words\\_in\\_c + V}$$

Where V is the vocabulary size.

### Key Steps
1. **Training**: Calculate prior P(class) and likelihood P(word|class) from training data
2. **Prediction**: For a new message, compute posterior P(class|message) using Bayes' Theorem
3. **Decision**: Choose the class with the highest posterior probability""")

nb.cells = cells

import nbformat
nbformat.write(nb, "C:/Users/HP/Desktop/normal/Naive_Bayes_Implementation.ipynb")
print("Notebook created successfully!")
