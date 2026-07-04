# ML Notes 6 — Unsupervised Learning & NLP

## Unsupervised Learning

Unsupervised learning is a type of machine learning in which a model learns patterns, structures, or relationships from data without being given labeled answers or target outcomes.

## Clustering

Clustering is the process of dividing a dataset into groups (clusters) based on similarity or distance measures without using predefined labels.

### K-Means Clustering

It's a method to divide a dataset into K clusters, where each data point belongs to the cluster with the nearest center called a centroid. An optimum value of K is obtained through the elbow method.

**Elbow Method:** The Elbow Method is a graphical technique for selecting the optimal number of clusters in K-Means by identifying the point at which increasing the number of clusters yields diminishing reductions in WCSS (Within-Cluster Sum of Squares).

### Hierarchical Clustering

Hierarchical clustering is an unsupervised clustering technique that builds a hierarchy of clusters by either repeatedly merging smaller clusters (**agglomerative**) or repeatedly splitting larger clusters (**divisive**), often visualized using a dendrogram.

Unlike **K-Means**, it does not require specifying the number of clusters beforehand.

A **dendrogram** is a tree diagram that shows how clusters are merged or split.

**Agglomerative Clustering (Bottom-Up Approach):**

- Starts with each data point as its own cluster.
- Repeatedly merges the most similar clusters.
- Continues until all points belong to a single cluster or a stopping criterion is met.

**Divisive Clustering (Top-Down Approach):**

- Starts with all data points in one cluster.
- Repeatedly splits clusters into smaller clusters.
- Continues until each point is in its own cluster or the desired number of clusters is reached.

### DBSCAN (Density Based Spatial Clustering of Applications with Noise)

**DBSCAN** is a clustering algorithm that groups together points that are closely packed and marks isolated points as outliers (noise).

**i) Epsilon (ε):** The maximum distance within which points are considered neighbors.

**ii) MinPts (Minimum Points):** The minimum number of points required within the ε-neighborhood to form a dense region.

**iii) Core Point:** A point is a core point if it has at least MinPts points (including itself) within its ε-neighborhood.

> Example: If ε = 2 and MinPts = 5, a point having 5 or more points within distance 2 is a core point.

**iv) Border Point:** Has fewer than MinPts neighbors but is reachable from a core point.

**v) Noise Point:** Neither a core point nor reachable from any core point.

> When we have a dataset which has too much volume of data or a lot of density of the data, with lots of datapoints in one place, we can use DBSCAN — otherwise we can use K-Means or Hierarchical clustering.

### Evaluation of Clusters

**1. External Evaluation**

- Measures how well the clusters match the true labels or known classes.
- Requires ground-truth data.
- *Example:* Comparing clustered students with their actual departments.

**2. Internal Evaluation**

- Measures the quality of clusters using only the data itself, without true labels.
- Checks whether points within a cluster are similar and clusters are well separated.
- *Example:* Evaluating how compact and distinct the clusters are.

**3. Reliability Evaluation**

- Measures how consistent the clustering results are when the algorithm is run multiple times or the data changes slightly.
- *Example:* Getting similar clusters in repeated runs of K-Means.

## Natural Language Processing (NLP)

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and generate human language.

### 1. Tokenization

The process of breaking text into smaller units called tokens (words, sentences, or phrases).

*Example:* "Machine learning is fun" → ["Machine", "learning", "is", "fun"]

### 2. Part-of-Speech (POS) Tagging

Identifying the grammatical role of each word in a sentence (noun, verb, adjective, etc.).

*Example:* "She runs fast"

- She → Pronoun
- runs → Verb
- fast → Adverb

### 3. Stop Words

Common words that are often removed because they carry little meaning.

*Examples:* "the", "is", "and", "in", "of"

### 4. Stemming

Reduces a word to its root form by removing prefixes or suffixes, sometimes producing non-dictionary words.

*Example:*

- Playing → Play
- Studies → Studi

### 5. Lemmatization

Reduces a word to its dictionary (base) form while considering its meaning and grammar.

*Example:*

- Running → Run
- Better → Good

### 6. Named Entity Recognition (NER)

Identifies and classifies named entities such as people, organizations, locations, dates, etc., in text.

*Example:* "Apple released a new iPhone."

- Apple → Organization
- iPhone → Product

NER helps distinguish meanings. For example:

- "Apple is a fruit." — Apple = Fruit (not an organization)
- "Apple launched a phone." — Apple = Company

**BIO Tagging in NER**

BIO tagging labels words based on their position in an entity.

- **B (Beginning):** First word of an entity.
- **I (Inside):** Continuation of the same entity.
- **O (Outside):** Not part of any entity.

### Bag of Words (BoW)

A method that represents text by counting how many times each word appears, ignoring grammar and order.

### TF-IDF (Term Frequency – Inverse Document Frequency)

A method that gives importance to words based on how often they appear in a document and how rare they are across all documents.

### N-grams

A way of representing text using sequences of N words together.

- **Bigram (2-gram):** 2 words together — *Example: "machine learning"*
- **Trigram (3-gram):** 3 words together — *Example: "natural language processing"*
- **Skip-gram:** Words that are related but not necessarily next to each other.

### CBOW (Continuous Bag of Words)

A model that predicts a target word using surrounding words (context).

*Example:* "The cat ___ on the mat" → predicts "sat"

### Word2Vec

A technique that converts words into numerical vectors (embeddings) so that similar words have similar meanings in vector space.

*Example:* "king" and "queen" will be closer in meaning than "king" and "car".
