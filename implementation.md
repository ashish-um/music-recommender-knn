# Music Recommendation System using K-Nearest Neighbors (KNN)

**Mini Project: Artificial Intelligence & Machine Learning (PCC-302-ITT)**
Savitribai Phule Pune University, Third Year Information Technology (2024 Course)

**Notebook:** `Music_Recommendation_KNN.ipynb`
**Dataset:** Spotify Tracks Dataset (`dataset.csv`)
**Model:** K-Nearest Neighbors (scikit-learn)

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Background](#2-background)
   - 2.1 What is a recommendation system?
   - 2.2 Types of recommendation systems
   - 2.3 How a computer "hears" a song: Spotify audio features
   - 2.4 The K-Nearest Neighbors (KNN) algorithm
   - 2.5 Why feature scaling is necessary for KNN
   - 2.6 How do we measure whether recommendations are good?
   - 2.7 Cross validation and hyperparameter tuning
3. [Link to the Syllabus and Lab Assignments](#3-link-to-the-syllabus-and-lab-assignments)
4. [Tools, Files and How to Run](#4-tools-files-and-how-to-run)
5. [Implementation (step by step)](#5-implementation-step-by-step)
   - 5.1 Problem Definition
   - 5.2 Dataset Collection
   - 5.3 Data Preprocessing
   - 5.4 Model Development
   - 5.5 Performance Evaluation
   - 5.6 Model Optimization
   - 5.7 Result Analysis and Documentation
6. [Results](#6-results)
7. [Limitations and Future Work](#7-limitations-and-future-work)
8. [Glossary](#8-glossary)

---

## 1. Project Summary

This project builds a system that **recommends songs**. You give it the name of one song you like, and it returns the **10 songs that sound most similar** to it.

It does this using only **how the songs sound**: their energy, danceability, loudness, tempo, acousticness and other measurements published by Spotify. It does not use any information about the listener. This approach is called **content-based recommendation**.

The algorithm used is **K-Nearest Neighbors (KNN)** from the scikit-learn library. KNN is used in two ways:

| Use | scikit-learn class | Purpose |
|---|---|---|
| **Recommender** | `NearestNeighbors` | Finds the 10 songs closest to the given song. This is the actual recommendation system. |
| **Classifier** | `KNeighborsClassifier` | Predicts a song's **genre** from its audio features. It uses the same neighbour logic, but its output can be checked with standard metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix). |

The project also compares **Decision Tree** (`DecisionTreeClassifier`) and
**Gaussian Naïve Bayes** (`GaussianNB`) against KNN for genre prediction, following
PR7 and PR8. The comparison uses the same features and train/test rows, reports
accuracy, weighted precision, recall and F1, and identifies the highest test accuracy.
The recommender itself continues to use nearest-neighbour similarity.

The project follows the seven stages of a machine learning project: **Problem Definition, Dataset Collection, Data Preprocessing, Model Development, Performance Evaluation, Model Optimization, and Result Analysis**.

**Final result:** after tuning with `GridSearchCV`, genre-prediction accuracy improved from **57.56% to 61.83%**. On average, **about 5 of the 10 recommended songs** belong to the same genre as the input song.

---

## 2. Background

### 2.1 What is a recommendation system?

A **recommendation system** (or *recommender*) is a program that suggests items a user is likely to enjoy. Examples:

- YouTube suggests the next video.
- Amazon shows "customers who bought this also bought…".
- Spotify creates playlists such as *Discover Weekly* and *Song Radio*.

Music apps hold tens of millions of songs, far too many for anyone to browse. A recommender narrows this down to a handful of songs the listener will probably like.

In AI terms, a recommender is an **intelligent agent**. It **senses** (reads what the user selected), **thinks** (compares the selection with the catalogue), and **acts** (shows a list of suggestions). The syllabus lists the *Smart Recommendation System* as a case study in Unit I.

### 2.2 Types of recommendation systems

| Type | Idea | Example | Needs |
|---|---|---|---|
| **Content-based filtering** | Recommend items whose *properties* are similar to an item the user liked. | "You liked a calm piano piece, here are other calm piano pieces." | Data describing each item (here, audio features). |
| **Collaborative filtering** | Recommend items liked by *other users with similar taste*. | "People who listen to A also listen to B." | Large listening history of many users. |
| **Hybrid** | Combine both. | What Spotify and Netflix actually do. | Both kinds of data. |

This project uses **content-based filtering** because:

1. The dataset contains rich descriptions of each song but **no user history**.
2. It works even for a brand-new user or a brand-new song. Collaborative filtering cannot handle these cases, which is known as the *cold-start problem*.
3. It can be built entirely with scikit-learn algorithms from the syllabus.

### 2.3 How a computer "hears" a song: Spotify audio features

A computer cannot "listen" to music the way we do. Spotify analyses every song's audio and publishes numeric measurements called **audio features**. Each song becomes a row of numbers, which a machine learning model can work with.

The features used in this project:

| Feature | Range | Meaning |
|---|---|---|
| `danceability` | 0.0 – 1.0 | How suitable the track is for dancing (steady beat, rhythm stability). |
| `energy` | 0.0 – 1.0 | Intensity and activity. Metal is high; a soft piano piece is low. |
| `loudness` | ≈ −60 to 0 dB | Overall loudness of the track in decibels. |
| `speechiness` | 0.0 – 1.0 | Presence of spoken words. Rap scores higher than instrumental music. |
| `acousticness` | 0.0 – 1.0 | Confidence that the track is acoustic (not electronic). |
| `instrumentalness` | 0.0 – 1.0 | Likelihood that the track has no vocals. |
| `liveness` | 0.0 – 1.0 | Probability that the track was recorded live with an audience. |
| `valence` | 0.0 – 1.0 | Musical positiveness. High values sound happy; low values sound sad or angry. |
| `tempo` | beats per minute | Speed of the song. |
| `duration_ms` | milliseconds | Length of the song. |
| `mode` | 0 or 1 | Minor (0) or major (1) key. |
| `explicit` | True / False | Whether the lyrics are explicit. |

**Key idea:** if two songs have similar numbers for all of these features, they probably **sound similar**. This is the assumption the whole project is built on.

### 2.4 The K-Nearest Neighbors (KNN) algorithm

#### 2.4.1 Intuition

Imagine plotting every song as a point on a graph, for example with `energy` on one axis and `acousticness` on the other. Calm acoustic songs would gather in one corner and loud electronic songs in another. Songs that are **close together on this graph sound alike**.

With 12 features, every song is a point in a **12-dimensional space**. We cannot draw this, but the mathematics of measuring distance works the same way.

KNN simply says:

> To make a decision about a new point, look at the **K points closest to it** and use them.

#### 2.4.2 KNN for recommendation (`NearestNeighbors`)

1. Represent every song as a vector of 12 numbers.
2. When the user picks a song, compute the distance from that song to every other song.
3. Sort by distance and return the closest K songs (here K = 10).

No training happens beyond storing the songs, which is why KNN is called a **lazy learner**.

#### 2.4.3 KNN for classification (`KNeighborsClassifier`)

To predict the genre of a song:

1. Find its K nearest songs in the training data.
2. Look at their genres.
3. Predict the genre that appears most often among them (a "majority vote").

For example, with K = 5, if the 5 nearest songs are *metal, metal, metal, edm, metal*, the prediction is **metal**.

#### 2.4.4 Distance metrics

How "closeness" is measured matters. Three metrics are tested in this project. For two songs **a** and **b** with features a₁…aₙ and b₁…bₙ:

| Metric | Formula | Explanation |
|---|---|---|
| **Euclidean** (default) | √( Σ (aᵢ − bᵢ)² ) | Straight-line distance ("as the crow flies"). |
| **Manhattan** | Σ \|aᵢ − bᵢ\| | Sum of absolute differences ("city-block" distance). Less affected by one very large difference. |
| **Cosine** | 1 − (a · b) / (‖a‖ ‖b‖) | Compares the *direction* of two vectors (the overall profile), not their length. |

#### 2.4.5 Important hyperparameters

A **hyperparameter** is a setting chosen by us before training, not learned from data.

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `n_neighbors` (K) | Number of neighbours that vote. | A small K is sensitive to noise (overfitting). A large K gives smoother decisions but may ignore local detail (underfitting). |
| `weights` | `uniform`: every neighbour has an equal vote. `distance`: closer neighbours have a bigger vote (weight = 1 / distance). | `distance` usually helps when K is large. |
| `metric` | Distance formula (Euclidean, Manhattan, Cosine). | Changes which songs count as "close". |

#### 2.4.6 Advantages and disadvantages of KNN

| Advantages | Disadvantages |
|---|---|
| Very simple to understand and explain. | Slow on very large datasets, because distance is computed to every point. |
| No training phase. | Sensitive to feature scale (Section 2.5). |
| Naturally gives "most similar items", which is exactly what a recommender needs. | Performance drops when there are many irrelevant features. |
| Works for multi-class problems (7 genres). | The value of K must be chosen carefully. |

### 2.5 Why feature scaling is necessary for KNN

The features have very different ranges:

- `danceability` lies between 0 and 1.
- `tempo` lies between about 0 and 250.
- `duration_ms` lies between about 17,000 and 2,600,000.

Without scaling, a difference of 30 seconds in duration (30,000 ms) would completely outweigh any difference in danceability (at most 1). KNN would end up recommending songs **of the same length**, not songs that **sound the same**.

**Min-Max scaling** converts every feature to the range 0 to 1:

```
x_scaled = (x − min) / (max − min)
```

After scaling, every feature has an equal chance to influence the distance.

### 2.6 How do we measure whether recommendations are good?

This is the main design challenge of the project. In a normal classification problem (like Titanic survival), the dataset contains the correct answer. For a recommender there is **no column saying "this is the correct recommendation"**, because "good" depends on the listener's taste.

The project solves this with a commonly used **proxy (substitute)**: the song's **genre**. The reasoning is that if the system has truly learned what songs sound like, then:

1. It should be able to **predict a song's genre** from its sound. This is measured with the classifier.
2. Most of the songs it **recommends** should belong to the **same genre** as the input song. This is measured with Precision@10.

#### 2.6.1 Classification metrics

| Metric | Formula | Meaning in this project |
|---|---|---|
| **Accuracy** | correct predictions / total predictions | Percentage of test songs whose genre was predicted correctly. |
| **Precision** | TP / (TP + FP) | Of the songs predicted as "jazz", how many really are jazz. |
| **Recall** | TP / (TP + FN) | Of all real jazz songs, how many were found. |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance of Precision and Recall. |
| **Confusion Matrix** | Table of actual vs predicted classes | Shows *which* genres get confused with which. |

TP = True Positive, FP = False Positive, FN = False Negative.

Since there are 7 genres, Precision, Recall and F1 are calculated for each genre and then combined using a **weighted average**, where each genre counts in proportion to its number of songs (`average="weighted"`).

#### 2.6.2 Recommendation metric: Precision@K

```
Precision@10 = (number of the 10 recommended songs with the same genre as the input song) / 10
```

This value is calculated for every song in the test set and then averaged. A Precision@10 of 0.50 means that on average **5 of the 10 recommended songs share the input song's genre**. For comparison, random recommendations from 7 genres would score only about 0.14.

### 2.7 Cross validation and hyperparameter tuning

#### Cross validation

Testing a model on a single train/test split can be lucky or unlucky. **K-Fold Cross Validation** (here 5-fold) works as follows:

1. Split the training data into 5 equal parts (folds).
2. Train on 4 folds and test on the 5th.
3. Repeat 5 times so that every fold is used once for testing.
4. Report the **average** accuracy (and its standard deviation).

This gives a more reliable estimate of performance.

#### Grid Search

**`GridSearchCV`** tries **every combination** of the hyperparameter values we provide. It evaluates each combination with 5-fold cross validation and keeps the best one.

In this project: 9 values of K × 2 weights × 3 metrics = **54 combinations**, each tested 5 times, for **270 model fits**.

---

## 3. Link to the Syllabus and Lab Assignments

Every technique used in the project is part of the course:

| Project component | Syllabus unit | Lab assignment |
|---|---|---|
| Intelligent agent + PEAS representation, *Smart Recommendation System* case study | Unit I | PR1 |
| Data cleaning, missing values, encoding, feature scaling, EDA (histograms, boxplots), train-test split | Unit III | PR5 |
| K-Nearest Neighbors classification | Unit III | PR7 |
| Decision Tree and Naïve Bayes classification; three-model comparison | Unit III | PR8 |
| Accuracy, Precision, Recall, F1-Score, Confusion Matrix | Unit III | PR7, PR8 |
| Cross Validation, Hyperparameter Tuning with Grid Search, before vs after comparison | Unit V | PR11 |

The notebook is also written in the same style as the lab notebooks:
- the "Developed by / Checked by" cell
- headings such as `#### LOADING DATASET` and `#### TRAIN TEST SPLIT`
- the same missing-value printing loop
- a grouped bar chart comparing metrics
- a printed "RESULT ANALYSIS" block

---

## 4. Tools, Files and How to Run

### 4.1 Software

| Tool | Use |
|---|---|
| Python 3 | Programming language |
| Jupyter Notebook | Running the code cell by cell |
| pandas | Loading and cleaning the table of songs |
| NumPy | Numerical arrays |
| Matplotlib, Seaborn | Graphs (histograms, boxplots, heatmaps, bar charts) |
| scikit-learn | `LabelEncoder`, `MinMaxScaler`, `train_test_split`, `KNeighborsClassifier`, `NearestNeighbors`, `cross_val_score`, `GridSearchCV`, metrics |

Install, if needed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

### 4.2 Files

```
project-folder/
├── Music_Recommendation_KNN.ipynb   ← the notebook
├── dataset.csv                      ← Spotify Tracks Dataset (download separately)
└── music_recommender_gui/            ← PyQt6 app, reusable ML core and tests
```

### 4.3 Getting the dataset

1. Open https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset. The same dataset is also on Kaggle as "Spotify Tracks Dataset" by Maharshi Pandya.
2. Download **`dataset.csv`** (about 20 MB).
3. Place it in the **same folder** as the notebook.

### 4.4 Running

Open the notebook in Jupyter and choose **Kernel → Restart & Run All**. The whole notebook runs in well under a minute on a normal laptop; the Grid Search takes a few seconds. All results are reproducible because `random_state=42` is used for the split.

---

## 5. Implementation (step by step)

This section follows the notebook in order. Each step explains **what** is done, **why**, the **code**, and the **output** obtained.

### 5.1 Problem Definition

**Problem:** Given one song a listener likes, recommend 10 other songs that sound similar, using only the songs' audio features.

**Objectives:**
1. Prepare a clean, scaled dataset of songs.
2. Build a KNN-based recommender.
3. Evaluate it objectively using genre as ground truth.
4. Improve it using Cross Validation and Grid Search.
5. Analyse where it works well and where it does not.

**PEAS Representation** of the recommendation agent:

| PEAS Component | Music Recommendation System |
|---|---|
| **Performance Measure** | Accuracy, Precision, Recall, F1-Score of genre prediction; Precision@10 of recommendations |
| **Environment** | The song catalogue (Spotify Tracks Dataset) and the user selecting a song in a Jupyter Notebook |
| **Actuators** | Displaying the list of top-10 recommended songs (track name, artist, genre, distance) |
| **Sensors** | The song name entered by the user and that song's audio features from the dataset |

### 5.2 Dataset Collection

#### About the dataset

| Property | Value |
|---|---|
| Name | Spotify Tracks Dataset |
| Source | Collected from the Spotify Web API, published on Hugging Face and Kaggle |
| License | BSD |
| Rows (songs) | 114,000 |
| Columns | 21 |
| Genres | 114 (exactly 1,000 songs each) |

Columns include identifiers (`track_id`, `artists`, `album_name`, `track_name`), `popularity`, the audio features described in Section 2.3, `key`, `time_signature` and the label `track_genre`.

#### Importing libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
```

#### Loading the dataset

```python
df = pd.read_csv("dataset.csv")

print("Number of samples: ", df.shape[0])
print("Number of features: ", df.shape[1])
```

Output:
```
Number of samples:  114000
Number of features:  21
```

The notebook then shows the first 5 rows with `df.head(5)`. It prints every column's name, number of non-empty values and data type using the same loop as the lab Data Preprocessing assignment. It also prints the number of genres (114).

### 5.3 Data Preprocessing

#### Step 1: Removing a useless column

`Unnamed: 0` is just a saved row number with no meaning, so it is dropped.

```python
df.drop(columns=["Unnamed: 0"], inplace=True)
```

#### Step 2: Detecting missing values

```python
for col in df.columns:
    print(f"Column Name      : {col}")
    print(f"Missing Values   : {df[col].isnull().sum()}")
    print("-" * 35)
```

Result: `artists`, `album_name` and `track_name` each have **1 missing value**, all in the same row. Every other column is complete.

#### Step 3: Handling missing values

A song with no name or artist cannot be shown as a recommendation, and it is only 1 row out of 114,000. So the row is **removed** instead of filled in.

```python
df.dropna(inplace=True)
```

Output: `Missing values after handling: 0`, `Number of samples: 113999`.

#### Step 4: Selecting genres

114 genres is too many for a mini project, and many overlap heavily (for example *rock*, *alt-rock*, *hard-rock*, *punk-rock*). With all of them, the evaluation results would be very low and hard to interpret. So **7 clearly different genres** are selected:

```python
selected_genres = ["classical", "hip-hop", "heavy-metal", "edm", "jazz", "country", "pop"]
df = df[df["track_genre"].isin(selected_genres)].reset_index(drop=True)
```

Output: 7,000 songs (1,000 per genre).

#### Step 5: Handling duplicate songs

Two kinds of duplicates exist:

1. **Same `track_id` under different genres.** The exact same Spotify track is listed, for example, under both *pop* and *edm*.
2. **Same song released more than once.** The same name and artist appear on a single, an album and a compilation, each with a different `track_id`.

If duplicates were kept, the system would recommend **the same song back to the user**. They would also inflate the results, because a copy of a test song could sit in the training set.

```python
df = df.drop_duplicates(subset="track_id")
df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
```

Output:
```
Duplicate track_id: 255
Duplicate songs (same name and artist): 1820
Number of samples after removing duplicates: 4925
```

Songs per genre after cleaning:

| Genre | Songs |
|---|---|
| heavy-metal | 986 |
| classical | 815 |
| hip-hop | 727 |
| edm | 674 |
| pop | 644 |
| country | 607 |
| jazz | 472 |
| **Total** | **4,925** |

#### Step 6: Statistical analysis

`df.describe()` shows the count, mean, standard deviation, minimum, quartiles and maximum of every numeric column. For example:
- the mean `energy` is about 0.60
- the median `acousticness` is only 0.18, but the mean is 0.35, which means a few very acoustic genres (classical, jazz) pull the average up
- `duration_ms` has a maximum of about 2.6 million ms (≈ 44 minutes), which is an extreme outlier

#### Step 7: Data visualisation

| Plot | What it shows |
|---|---|
| **Count plot** of songs per genre | The classes are reasonably balanced after cleaning; jazz is the smallest. |
| **Histograms** of the 9 audio features | Distribution shapes. For example, `instrumentalness` and `speechiness` are heavily skewed towards 0, while `acousticness` has two peaks (electronic vs acoustic music). |
| **Boxplots** of energy, acousticness and danceability **by genre** | Classical has very low energy and very high acousticness. Heavy-metal has very high energy and almost zero acousticness. Hip-hop is the most danceable. Pop, country and EDM overlap a lot, which already hints that these genres will be harder to separate. |
| **Boxplot** of duration | Several very long tracks (outliers). |
| **Correlation heatmap** | `energy` and `loudness` are strongly positively correlated, and `energy` and `acousticness` are strongly negatively correlated. |

#### Step 8: Encoding

Machine learning models need numbers, not text.

```python
encoder = LabelEncoder()
df["explicit"] = encoder.fit_transform(df["explicit"])          # False → 0, True → 1

genre_encoder = LabelEncoder()
df["genre_label"] = genre_encoder.fit_transform(df["track_genre"])
```

Genre labels: `classical = 0, country = 1, edm = 2, heavy-metal = 3, hip-hop = 4, jazz = 5, pop = 6`.

`mode` is already 0/1, so it needs no encoding.

#### Step 9: Choosing input features (X) and target (y)

```python
feature_cols = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo",
                "duration_ms", "mode", "explicit"]

X = df[feature_cols]
y = df["genre_label"]
```

Columns that were **deliberately left out**:

| Column | Reason |
|---|---|
| `track_id`, `artists`, `album_name`, `track_name` | Text identifiers. They are only used to display recommendations. |
| `popularity` | Describes how *famous* a song is, not how it *sounds*. A content-based recommender should match sound. |
| `key`, `time_signature` | Including them gave lower accuracy during testing (they add noise to the distance). |
| `track_genre` | It is the target, so using it as an input would be cheating. |

#### Step 10: Train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- **80%** training (3,940 songs) and **20%** testing (985 songs).
- `stratify=y` keeps the same genre proportions in both sets.
- `random_state=42` makes the split the same every time.

#### Step 11: Feature scaling

```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler learns the minimum and maximum **from the training data only** (`fit_transform`) and applies the same values to the test data (`transform`). This prevents information from the test set from "leaking" into training. After scaling, all 12 features lie between 0 and 1 (Section 2.5).

### 5.4 Model Development

#### 5.4.1 Default KNN classifier

```python
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
```

Default parameters are `n_neighbors=5`, `weights="uniform"` and `metric="minkowski"` with `p=2`, which is Euclidean distance.

#### 5.4.2 KNN recommender

```python
all_songs_scaled = scaler.transform(X)

recommender = NearestNeighbors(n_neighbors=11, metric="euclidean")
recommender.fit(all_songs_scaled)
```

The recommender is fitted on **all 4,925 songs** so that any song in the catalogue can be searched. It asks for 11 neighbours because the closest "neighbour" of a song is always **the song itself** (distance 0), which is then removed.

#### 5.4.3 The recommendation function

```python
def recommend_songs(song_name, model, n_recommendations=10):
    matches = df[df["track_name"].str.lower() == song_name.lower()]

    if matches.empty:
        print(f"'{song_name}' is not present in the dataset.")
        return None

    song_index = matches.index[0]
    song = df.loc[song_index]
    print(f"Input Song : {song['track_name']}  |  Artist: {song['artists']}  |  Genre: {song['track_genre']}")

    distances, indices = model.kneighbors(all_songs_scaled[[song_index]], n_neighbors=n_recommendations + 1)

    result = df.loc[indices[0], ["track_name", "artists", "track_genre"]].copy()
    result["distance"] = distances[0].round(4)
    result = result[result.index != song_index].head(n_recommendations)

    return result.reset_index(drop=True)
```

How it works:
1. Find the song by name. The search is not case-sensitive.
2. If it is not found, print a message instead of crashing.
3. Take that song's scaled feature vector.
4. Ask the model for the nearest songs.
5. Remove the input song and return a table of track name, artist, genre and distance.

A smaller distance means a more similar song.

**Example:** `recommend_songs("Nuvole Bianche", recommender)`, a calm piano piece by Ludovico Einaudi. **All 10 recommendations are classical**: Mozart piano concertos, Handel, Verdi, Ravel and another Einaudi piece ("Una Mattina").

### 5.5 Performance Evaluation

#### 5.5.1 Genre prediction (default KNN)

```python
y_pred = knn.predict(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
```

| Metric | Default KNN |
|---|---|
| Accuracy | 0.5756 |
| Precision | 0.5723 |
| Recall | 0.5756 |
| F1-Score | 0.5684 |

The notebook also prints the full `classification_report` (per-genre precision, recall and F1) and draws the **confusion matrix** as a heatmap.

#### 5.5.2 Recommendation quality: Precision@10

```python
def precision_at_k(model, k=10):
    model.fit(X_train_scaled)
    distances, indices = model.kneighbors(X_test_scaled, n_neighbors=k)

    train_genres = y_train.values
    test_genres = y_test.values

    hits = (train_genres[indices] == test_genres[:, None]).sum(axis=1)
    return (hits / k).mean()
```

For each of the 985 test songs, the function:
1. finds its 10 nearest **training** songs
2. counts how many share its genre
3. divides by 10

It then averages over all test songs. Using training songs as the "catalogue" and test songs as "queries" means the songs being scored were never seen by the model.

Default result: **Precision@10 = 0.4936**.

#### 5.5.3 Choosing K

The classifier is trained for every K from 1 to 30 and the test accuracy is plotted. Accuracy is lowest for very small K (about 0.55–0.56 at K = 1–2, where a single noisy neighbour decides). It rises quickly and then levels off between about 0.585 and 0.594 from K ≈ 11 onwards. The best K in this graph is **17 (accuracy 0.5939)**. This graph gives an intuition for K, and the final choice is made properly with Grid Search in the next step.

### 5.6 Model Optimization

#### 5.6.1 Cross validation of the default model

```python
scores = cross_val_score(KNeighborsClassifier(), X_train_scaled, y_train, cv=5, scoring="accuracy")
```

```
Fold accuracies: [0.566  0.5711 0.5622 0.5635 0.5698]
Mean CV Accuracy: 56.65 %
CV Standard Deviation: 0.0035
```

The five folds give very similar scores (standard deviation 0.0035), so the default model's performance is **stable**. The test accuracy (57.56%) agrees with it.

#### 5.6.2 Hyperparameter tuning with GridSearchCV

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25, 31],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"]
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_knn = grid.best_estimator_
```

```
Best Parameters: {'metric': 'manhattan', 'n_neighbors': 25, 'weights': 'distance'}
Best CV Accuracy: 62.01 %
```

Top 5 combinations:

| K | weights | metric | Mean CV accuracy |
|---|---|---|---|
| 25 | distance | manhattan | 0.6201 |
| 21 | distance | manhattan | 0.6188 |
| 31 | distance | manhattan | 0.6173 |
| 15 | distance | manhattan | 0.6142 |
| 11 | distance | manhattan | 0.6089 |

**Interpretation:**
- **Manhattan distance** wins clearly. Every one of the top 5 uses it, because it is less affected by one feature differing a lot.
- **Distance weighting** wins: when 25 neighbours vote, the closest ones should count more.
- A **larger K (around 25)** works better than the default 5, because the genre borders are fuzzy and more neighbours give a more stable vote.

#### 5.6.3 Evaluating the optimized model

| Metric | Optimized KNN |
|---|---|
| Accuracy | 0.6183 |
| Precision | 0.6160 |
| Recall | 0.6183 |
| F1-Score | 0.6125 |

A new confusion matrix is drawn for the optimized model.

#### 5.6.4 Optimized recommender

The best distance metric found by Grid Search (Manhattan) is also applied to the recommender:

```python
best_metric = grid.best_params_["metric"]
tuned_p_at_10 = precision_at_k(NearestNeighbors(metric=best_metric), k=10)

best_recommender = NearestNeighbors(n_neighbors=11, metric=best_metric)
best_recommender.fit(all_songs_scaled)
```

Optimized result: **Precision@10 = 0.5048**.

`n_neighbors` and `weights` affect only the *voting* in classification. The recommender always returns exactly 10 songs and does not vote, so only the **metric** carries over to it.

### 5.7 Result Analysis and Documentation

#### Comparison of three genre classifiers (PR7 and PR8)

The notebook's **COMPARISON: KNN, DECISION TREE AND NAÏVE BAYES** section and
the desktop app's **Results → Model comparison** evaluate:

| Classifier | Configuration |
|---|---|
| KNN | Grid-search optimized configuration; configured baseline if grid search is disabled in the app |
| Decision Tree | `DecisionTreeClassifier(random_state=42)` with other defaults; the app uses its configured seed |
| Naïve Bayes | `GaussianNB()` with defaults |

All three use the same cleaned songs, 12 default input features, stratified 80/20
split and training-fitted Min-Max scaler. App settings apply to all three when
changed. Each model fits only the training songs and predicts the test genres.
The comparison table shows accuracy and weighted precision, recall and F1, sorted
by accuracy. A bar chart shows accuracy as a percentage. The highest test accuracy
is named, including ties. Confusion matrices are available for all three models;
the app also provides per-genre reports for each.

An end-to-end notebook run with the default dataset and settings produced:

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---|---|---|---|
| **KNN (optimized)** | **0.6183** | **0.6160** | **0.6183** | **0.6125** |
| Decision Tree | 0.5604 | 0.5607 | 0.5604 | 0.5603 |
| Gaussian Naïve Bayes | 0.5513 | 0.5294 | 0.5513 | 0.5186 |

**KNN has the highest test accuracy (61.83%) among these tested configurations.**
Run the notebook or retrain in the app to recompute scores after changing settings.
Existing saved models do not gain comparison data automatically; retrain and save
a new model. Comparison scores and reports persist with the model bundle.

This identifies the **best tested configuration on this test set**, not the best
model in general. Only KNN is grid-search tuned here. For further model selection,
compare training-set cross-validation scores and reserve a test set for final
evaluation. A classifier's accuracy does not measure recommendation quality:
recommendations still use `NearestNeighbors` and are evaluated by Precision@10.

#### 5.7.1 Before vs after optimization

| Metric | Default KNN | Optimized KNN | Improvement |
|---|---|---|---|
| Accuracy | 0.5756 | 0.6183 | +0.0426 |
| Precision | 0.5723 | 0.6160 | +0.0437 |
| Recall | 0.5756 | 0.6183 | +0.0426 |
| F1-Score | 0.5684 | 0.6125 | +0.0442 |
| CV Accuracy | 0.5665 | 0.6201 | +0.0536 |
| Precision@10 | 0.4936 | 0.5048 | +0.0112 |

The notebook shows this as a grouped bar chart, in the same style as the PR11 lab assignment.

#### 5.7.2 Genre-wise performance (optimized model)

| Genre | Precision | Recall | F1-Score |
|---|---|---|---|
| classical | 0.96 | 0.79 | **0.86** |
| heavy-metal | 0.78 | 0.93 | **0.85** |
| hip-hop | 0.59 | 0.57 | 0.58 |
| edm | 0.47 | 0.56 | 0.51 |
| country | 0.47 | 0.53 | 0.50 |
| jazz | 0.51 | 0.46 | 0.48 |
| pop | 0.34 | 0.24 | **0.28** |

#### 5.7.3 Confusion matrix (optimized model)

Rows are the actual genre and columns are the predicted genre.

|  | classical | country | edm | heavy-metal | hip-hop | jazz | pop |
|---|---|---|---|---|---|---|---|
| **classical** | **128** | 3 | 0 | 1 | 0 | 21 | 10 |
| **country** | 0 | **64** | 11 | 14 | 12 | 12 | 8 |
| **edm** | 0 | 16 | **75** | 22 | 13 | 0 | 9 |
| **heavy-metal** | 0 | 8 | 4 | **184** | 0 | 0 | 1 |
| **hip-hop** | 0 | 8 | 29 | 4 | **83** | 2 | 19 |
| **jazz** | 6 | 16 | 7 | 5 | 3 | **44** | 14 |
| **pop** | 0 | 21 | 33 | 6 | 30 | 8 | **31** |

The diagonal (bold) shows correct predictions: 609 of 985 songs.

#### 5.7.4 Sample recommendations

| Input song | Genre | What was recommended |
|---|---|---|
| *Nuvole Bianche*, Ludovico Einaudi | classical | 10/10 classical: Mozart, Chopin, Handel, Verdi, Einaudi's "Una Mattina". |
| *Balls to the Wall*, Accept | heavy-metal | 5 heavy-metal, plus hard-rock-sounding tracks labelled country (ZZ Top "Sharp Dressed Man", Lynyrd Skynyrd "Simple Man"). |
| *Super Freaky Girl*, Nicki Minaj | hip-hop | A mix of hip-hop (DaBaby, Machine Gun Kelly), EDM (Skrillex remix of Kendrick Lamar's "HUMBLE.") and country. |
| *September*, Earth, Wind & Fire | jazz | Mostly upbeat country and rock (Toby Keith, Bachman-Turner Overdrive). |
| *Some Unknown Song* | n/a | Prints "'Some Unknown Song' is not present in the dataset." (error handling works) |

---

## 6. Results

### 6.1 Key findings

1. **Optimization helped every metric.** Grid Search raised accuracy by about **4.3 percentage points** (57.56% → 61.83%) and cross-validated accuracy by about **5.4 points** (56.65% → 62.01%). The best configuration is **K = 25, Manhattan distance, distance-weighted voting**.

2. **The recommender is far better than random.** About **5 of every 10** recommended songs match the input song's genre. A random recommender over 7 genres would match only about 1.4 of 10.

3. **Genres with a distinctive sound work very well.**
   - Classical (very low energy, very high acousticness) has F1 = 0.86.
   - Heavy-metal (very high energy, near-zero acousticness) has F1 = 0.85; 184 of 197 metal songs were identified.
   - The boxplots in Section 5.3 show why: these two genres sit at opposite extremes of the feature space.

4. **Pop is the hardest genre (F1 = 0.28).** The confusion matrix shows pop songs are most often predicted as **EDM (33)**, **hip-hop (30)** and **country (21)**. This is realistic: modern pop borrows electronic beats, rap verses and country styles, so in terms of *sound* it genuinely overlaps with those genres.

5. **Classical and jazz are sometimes confused (21 classical songs predicted as jazz).** Both are acoustic, often instrumental and calm.

6. **Precision@10 improved only slightly (0.49 → 0.50).** This is expected. Only the distance metric transfers from the classifier to the recommender; K and the weighting affect only the classifier's voting.

### 6.2 Is 62% accuracy good?

For this problem, yes. Genre is a **human-assigned label**, partly based on culture, artist and marketing, not only on sound. Two songs with almost identical audio features can be labelled *pop* and *edm*. The 12 audio features cannot fully capture the difference, so perfect accuracy is impossible from audio alone. The relevant comparison is against chance (about 14% for 7 genres), and the model is **more than 4× better than chance**.

More importantly, for a **recommender** a "wrong genre" recommendation is not necessarily a bad recommendation. ZZ Top's "Sharp Dressed Man" sounds like hard rock and is a reasonable suggestion for a heavy-metal listener, even though the dataset labels it *country*.

---

## 7. Limitations and Future Work

### Limitations

| Limitation | Explanation |
|---|---|
| **Sound only** | The system knows nothing about lyrics, language, mood the user wants, or the artist's fame. |
| **No personalisation** | Every user who picks the same song gets the same 10 recommendations. |
| **Noisy genre labels** | Some labels are questionable. For example, some Bollywood songs ("Zara Zara", "Kajra Re") are labelled *classical*, which lowers measured accuracy. |
| **Genre as a proxy** | Using genre to measure recommendation quality is an approximation; real quality can only be judged by listeners. |
| **7 genres only** | Results with all 114 genres would be much lower because of overlap. |
| **Duplicate names** | If two different songs share a name, `recommend_songs` uses the first one. |
| **Scalability** | KNN compares against every song. For millions of songs, approximate nearest-neighbour methods would be required. |

### Future work

1. **Hybrid recommender:** combine with collaborative filtering using real listening history.
2. **More features:** add lyrics (text features), language or release year.
3. **Feature weighting / selection:** let some features (e.g., energy, acousticness) count more in the distance.
4. **Dimensionality reduction:** apply PCA (Unit IV, PR10) to visualise songs in 2-D and possibly remove noise.
5. **Clustering:** use K-Means (Unit IV, PR9) to automatically discover "moods" or sub-genres.
6. **Search by artist as well as title** to resolve songs with the same name.
7. **User interface:** a small web app where the user types a song and sees recommendations.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Audio feature** | A number describing one property of a song's sound (e.g., energy). |
| **Content-based filtering** | Recommending items similar in their properties to an item the user liked. |
| **Collaborative filtering** | Recommending items liked by users with similar taste. |
| **Cold-start problem** | Collaborative filtering cannot recommend for new users/songs with no history. |
| **KNN** | K-Nearest Neighbors: decide using the K closest data points. |
| **Lazy learner** | An algorithm that does no real training and does its work at prediction time. |
| **Hyperparameter** | A setting chosen before training (e.g., K), not learned from data. |
| **Distance metric** | Formula used to measure how far apart two data points are. |
| **Min-Max scaling** | Rescaling a feature to the range 0–1. |
| **Label encoding** | Converting text categories into integers. |
| **Stratified split** | Splitting data so each class keeps the same proportion in train and test sets. |
| **Data leakage** | Accidentally letting test data influence training, which gives over-optimistic results. |
| **Cross validation** | Repeated train/test on different folds to get a reliable performance estimate. |
| **GridSearchCV** | Trying all combinations of given hyperparameters with cross validation and keeping the best. |
| **Confusion matrix** | Table of actual vs predicted classes. |
| **Precision@K** | Fraction of the top-K recommended items that are relevant (here: same genre). |
| **PEAS** | Performance measure, Environment, Actuators, Sensors: a way of describing an AI agent. |
