# EECS 731 Project 3 - Clustering
### Author: Jace Kline

## Requirements:
 Blockbuster or art film?
1. Set up a data science project structure in a new git repository in your GitHub account
2. Download the one of the MovieLens datasets from https://grouplens.org/datasets/movielens/
3. Load the data set into panda data frames
4. Formulate one or two ideas on how the combination of ratings and tags by users helps the data set to establish additional value using exploratory data analysis
5. Build one or more clustering models to determine similar movies to recommend using the other ratings and tags of movies by other users as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

## Results:
Our overarching goal was to create clustering models on the given movie datasets for the purposes of grouping similar movies together. In order to accomplish this goal, it was imperative that we performed feature engineering to provide quality grouping data to our chosen clustering models. We used both of the provided datasets in conjunction to create a dataset ideal for unambiguous clustering. The success of our data preparation was evident in the consistency of the clustering performed by the three separate models that we chose. All three models performed identically and grouped all data entries in the same manner. Despite the consistency across models, one downfall of our feature engineering efforts was that all our clustering models used produced only two cluster categories. In the future, an effort to diversify the model input data might prove beneficial in the effort to diversifying the output clusters. See the entire report [here](./notebooks/movies.md).
