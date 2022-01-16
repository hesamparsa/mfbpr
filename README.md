## Introduction
The major goal of recommender systems is to help users discover relevant items such as movies to watch, text to read or products to buy, so as to create a delightful user experience.

## Feedback Type
To learn the preference of users, the system shall collect feedback from them. The feedback can be either **explicit** or **implicit** [Hu et al., 2008]. For example, **IMDB** collects star ratings ranging from one to ten stars for movies. YouTube provides the thumbs-up and thumbs-down buttons for users to show their preferences. It is apparent that gathering explicit feedback requires users to indicate their interests proactively. Nonetheless, explicit feedback is not always readily available as many users may be reluctant to rate products.

## Getting the Data
The MovieLens dataset is hosted by the GroupLens website. Several versions are available. We will use the MovieLens 100K dataset [Herlocker et al., 1999]. This dataset is comprised of  100,000  ratings, ranging from 1 to 5 stars, from 943 users on 1682 movies. It has been cleaned up so that each user has rated at least 20 movies. Some simple demographic information such as age, gender, genres for the users and items are also available. We can download the ml-100k.zip and extract the u.data file, which contains all the  100,000  ratings in the csv format.

## The Matrix Factorization Model
Matrix factorization is a class of collaborative filtering models. Specifically, the model factorizes the user-item interaction matrix (e.g., rating matrix) into the product of two lower-rank matrices, capturing the low-rank structure of the user-item interactions. First, we implement the matrix factorization model in this notebook: 000MF.ipynb

## Bayesian Personalized Ranking

In the former sections, only explicit feedback was considered and models were trained and tested on observed ratings. There are two demerits of such methods: First, most feedback is not explicit but implicit in real-world scenarios, and explicit feedback can be more expensive to collect. Second, non-observed user-item pairs which may be predictive for users’ interests are totally ignored, making these methods unsuitable for cases where ratings are not missing at random but because of users’ preferences. Non-observed user-item pairs are a mixture of real negative feedback (users are not interested in the items) and missing values (the user might interact with the items in the future). We simply ignore the non-observed pairs in matrix factorization. Clearly, these models are incapable of distinguishing between observed and non-observed pairs and are usually not suitable for personalized ranking tasks.