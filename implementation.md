# Music Recommendation System using KNN, K-Means and Hierarchical Clustering

**AI/ML mini project · Savitribai Phule Pune University · TE Information Technology**

Developed by **Ashish Umrao (4315), Abhijeet Kumar (4301), Aryan Singh (4313)**
Checked by **Prof. Vaishali Ingle**

The executable report is [Music_Recommendation_KNN.ipynb](Music_Recommendation_KNN.ipynb).
The desktop and Streamlit interfaces share the training and recommendation core in
`music_recommender_gui/src/music_recommender/core/`. See the app README for launch commands.

## 1. Problem Definition

Given a song, recommend other songs with similar audio features and genre. Compare
KNN retrieval, K-Means clustering and Ward hierarchical clustering using sklearn.
All distance calculations use **Euclidean distance**.

The system is content based: each song is represented by its scaled audio features
and binary genre indicators. No listening history, ratings or personal preferences
are available in this dataset.

| PEAS component | This project |
|---|---|
| Performance | Recommendation genre consistency, number of results returned and runtime; separate audio-only genre-classification scores |
| Environment | Song catalogue and a listener selecting a song |
| Actuators | Display recommendations, plots and evaluation tables; show a YouTube search result |
| Sensors | Selected song, its audio measurements and its known genre |

## 2. Dataset Collection

Use the **Spotify Tracks Dataset**, supplied locally as `dataset.csv`.
Source: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset

The source contains **114,000 rows and 114 genre labels**. The default selection
is the previous **seven genres**: classical, hip-hop, heavy-metal, EDM, jazz, country
and pop. The notebook and both apps use this selection. You can still change the
selection; `TrainingConfig.genres=None` explicitly selects every source genre.

With the default selection, cleaning leaves **4,925 songs across seven genres**.
Missing rows and duplicate tracks/title-artist pairs are removed.

## 3. Data Preprocessing

1. Drop the saved row-number column.
2. Remove rows with missing values.
3. Select the seven default genres, or use a manually chosen selection.
4. Remove duplicate track IDs, then duplicate title/artist pairs, so a song is not
   repeatedly recommended through different catalogue entries.
5. Encode `explicit` as 0/1 and create numeric genre labels for classification targets.
6. Make a reproducible stratified 80/20 training/test split (`random_state=42`).
7. Fit the audio scaler on training rows only, then transform training/test rows.
8. Fit a genre `OneHotEncoder` on training rows, using the catalogue's known genre
   categories as the schema. Append these indicators to recommendation inputs.

Default audio features:

`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`,
`liveness`, `valence`, `tempo`, `duration_ms`, `mode`, `explicit`.

A genre is categorical. Encoding classical=0, jazz=1, pop=2 as a distance feature
would incorrectly make some genre pairs closer than others. One-hot encoding gives
each genre its own binary column. With otherwise identical audio features, two
different genres have an added Euclidean distance of **sqrt(2)**; matching genres
add zero. No extra genre weighting or hard genre filter is applied during retrieval.

For the default seven-genre catalogue, recommendations have **19 input columns**:
**12 scaled audio features + 7 genre indicators**. The separate genre classifier
has **12 audio inputs**. Giving that classifier its target genre as an input would
leak the answer and invalidate its scores.

The notebook includes descriptive statistics, histograms, genre counts, duration
boxplots and audio-feature correlations. Dense genre plots use a readable display
size or explicitly show the largest genres; training still uses the full catalogue.

## 4. Model Development

### KNN recommendations

`NearestNeighbors(metric="euclidean")` retrieves the closest songs using the combined
audio-and-genre vector. The selected song itself is excluded. Requested recommendation
count (for example, 10) is independent of classification neighbor count.

Euclidean distance is `sqrt(sum((a_i - b_i)**2))` over the combined feature columns.

### K-Means recommendations

`KMeans(n_clusters=..., n_init=10, random_state=42)` groups the combined feature vectors.
Assign a new song to the nearest cluster centroid, then rank songs within that cluster
by Euclidean distance. Fitted songs retain their original memberships.

### Hierarchical recommendations

`AgglomerativeClustering(n_clusters=..., linkage="ward", compute_distances=True)` builds
a hierarchy from the same combined feature vectors. Ward linkage merges groups using
Euclidean within-cluster variance.

Agglomerative clustering has no native prediction operation. For unseen songs, this
project explicitly assigns the nearest training-cluster centroid, then retrieves the
closest songs inside that cluster. The saved model includes the learned centroids,
catalogue memberships and dendrogram data.

### Separate KNN genre classification

`KNeighborsClassifier(metric="euclidean")` predicts genre from **audio only**, providing
the classification exercise and metrics from the syllabus. Its voting does not
produce the recommendation ranking. Clustering IDs are not genre predictions.

## 5. Performance Evaluation

### Recommendation comparison

All three methods use the same reproducible sample of at most **5,000 training
candidates** and **1,000 held-out test queries**, stratified where possible. This
bounds hierarchical memory use. The full cleaned catalogue remains available for
interactive recommendations; other songs are assigned to learned cluster centroids.

| Metric | Meaning |
|---|---|
| Precision@K | Average number of same-genre matches divided by the requested K |
| Coverage@K | Average number of returned recommendations divided by K |
| Fit seconds | Time to fit the selected estimator; excludes elbow-search diagnostics |
| Query ms/song | Mean retrieval time per test query, including assignment of unseen songs |

A small cluster can return fewer than K songs. Missing slots count as misses in
Precision@K and lower Coverage@K; results are not backfilled from other clusters.

**Genre is a recommendation input.** Precision@K is therefore a **genre-consistency
diagnostic**, not an independent test of recommendation quality. A score near 100%
is expected when the encoded genre dominates candidate separation. It does not prove
that a listener will like the tracks. Audio similarity, user ratings or listening
feedback would be needed for a broader assessment; the current app has no such labels.

### Genre-classification evaluation

Use held-out audio inputs for accuracy, weighted precision, recall, F1-score,
confusion matrices and per-genre reports. These scores evaluate a different task
and do not use genre indicators as input.

The notebook contains the executed scores for the seven default genres. The seven-genre regression
fixture in the tests intentionally stays smaller for fast checks; its scores must not
be presented as full-catalogue results.

## 6. Model Optimization

### KNN classification

Cross-validation evaluates default KNN. Grid search tests:

- Neighbor counts: 3, 5, 7, 9, 11, 15, 21, 25, 31.
- Voting weights: uniform and distance.
- Distance: Euclidean only.

There are **18 combinations evaluated over 5 folds = 90 fits**. The accuracy-by-K
plot is descriptive; the grid-search CV score selects the classifier configuration.
Changing classification K and weights does not change the recommendation distances.
Baseline and tuned retrieval therefore have the same recommendation Precision@K.

### Cluster count

In Streamlit, **leave Number of clusters blank** for automatic elbow detection.
Enter an integer to override it. In the notebook or Python config, use
`n_clusters=None` for automatic selection. The notebook's explicit worked example
may set a manual count instead.

Fit K-Means for counts 1–12 (bounded by distinct training vectors), normalize the
inertia curve and choose its largest bend away from the line joining the endpoints.
Both clustering methods use that count. If there is no detectable bend or fewer
than three curve points, use **2 clusters** and report the fallback. This geometric
heuristic does not guarantee an optimal grouping.

Results marks the count actually used, its selection source and, when different,
the detected elbow. The dendrogram shows the last 25 groups for readability.

## 7. Result Analysis and Documentation

Use the freshly executed notebook and each saved model's Results tab for measured
scores. Do not compare results from different genre selections, splits or feature
representations as though they were the same experiment.

Expected interpretations:

- Genre inputs deliberately favor recommendations with matching genre.
- KNN and clustering can return identical lists when all nearest songs are already
  inside the selected cluster.
- Full-catalogue genre classification is harder than classification among seven
  widely separated genres. The target labels also overlap and contain noise.
- Genre-consistency scores can increase substantially after adding genre inputs;
  this reflects the changed inputs rather than proof of better listening outcomes.
- Timings depend on dataset size, implementation and concurrent workload.

New model files save the genre encoder alongside the scaler, fitted retrieval state,
cluster centers, memberships, scores and plots. Existing audio-only models still open
with a retraining notice; retrain to add genre inputs and the selected catalogue. Models
trained with another distance require retraining with Euclidean distance.

| Project component | Lab assignment |
|---|---|
| Agent and PEAS | PR1 |
| Cleaning, scaling, encoding, EDA and train/test split | PR5 |
| KNN classification and classification metrics | PR7 |
| K-Means, hierarchical clustering, elbow curve and dendrogram | PR9 |
| Cross-validation and grid search | PR11 |
