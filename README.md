# MNIST dataset K-Means Clustering Code

## Algorithm Design

Using MNIST dataset

Using K-Means clustering
1. set several different values of K, set the distance function (the two-parametric difference of two vectors and the cosine similarity of two vectors are used to measure), set the vector representation of the images (directly flatten the 28 * 28 images into a 784-dimensional vector.)
2. in the whole dataset, get K random data, as the initial clustering center
3. Start iteration, and take the mean vector of each cluster as clustering center after each iteration, and calculate the difference between each clustering center and the previous clustering center (2-parametric), and stop the iteration when the distance difference is less than a certain threshold (set to 1).

In this experiment, four values of K, 10, 15, 20, and 25, were used to try the 2-paradigm and cos similarity to calculate the distance.
For each cluster, the label with the highest number is used as the prediction for this cluster.
Using accuracy score to determine the ability of the model

## Experimental results

The accuracy score on the training set

#### Use 2-parametric distance measurement

| K | accuracy score |
|--|--|
|10 |0.5965|
|15 |0.6614|
|20 |0.7359|
|25 |0.7531|

#### Use cos similarity

| k | accuracy score |
|--|--|
|10 |0.6019|
| 15 |0.7092|
|20 |0.7309|
| 25 |0.7716|

## Visualization with t-SNE

You can refer to the images in code directory.
