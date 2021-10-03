# Machine Learning Week 1

## Introduction

### What is Machine Learning?

Two difinitions are offered.

1. The field of study that gives computers the ability to learn without being explicitly programmed. -> by Arthur Samuel
2. A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. -> by Tom Mitchell

For example, the playing checkers program:

E: the experience of playing many games of checkers
T: the task of playing checkers
P: the probability that the program will win the next game

Machine Learing problem contains two broad categories: Supervised Learning and Unsupervised Learning

### Supervised Learning

The essential difference between Supervised Learning and Unsupervised Learning is the formal has a known data set. The data set has the input and output, and we know the relationship between them.

Supervised Learning have two categories: regression and classification.

In a regression problem, we are trying to predict results within a **continuous** function, but in a classfication problem, we are trying to predict results in a discrete output.

**Example:**

Suppose we have the dataset of the size of houses on the real estate market, the task is to predict their price. Price as a function of size is apparently continuous, so this is a regression problem.

We could turn this example into a classfication problem by instead making our output abount whether the house "sells for more or less than the asking price". Here we are classifying the houses based on price into two discrete categories.

### Unsupervised Learning

Unsupervised Learning, on the other hand, allow us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on releationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results. You can't really know if it is correct.

**Example:**

Clustering: Take a collection of 1000 essays written on the US Economy, and find a way to automatically group these essays into a small number that are somehow similatr or related by different variables, such as word frequency, sentence length, page count and so on.

Non-clustering: The "Cocktail Party Algorithm", which can find structure in messy data(such as the identification of individual voices and music from a mesh of sounds at a cocktail party).
