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

## Linear Regression with One Variable

### Model Representation

Recall that _regression problems_, we are taking input variables and trying to fit the output onto a _continuous_ expected result function.

Linear regression with one variable is also known as "univariate linear regression."

Univariate linear regression is used when you want to predict a **single output** value y from a **single input** value x. Obviously, we're doing **supervised learning**.

#### The Hypothesis Function

Our hypothesis function has the general form:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

By specifying the $\theta_0$ and $\theta_1$, we can get our estimated output $h_\theta(x)$.

Then, Apparently, every $h_\theta(x)$ is a hypothesis, and what we want to get is the best fit function.

### Cost Function

We can measure the accuracy of the hypothesis function by using a **cost function**. This takes an average of all the results of the hypothesis with inputs from x's compared to the actual output y's.

$$J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)^2$$

To break it apart, it is $\frac{1}{2}\bar{x}$ where $\bar{x}$ is the mean of the squares of $h_\theta(x_i)-y_i$, or the difference between the predicted value and the actual value.

This function also called the "Squared error function", or "Mean squared error". The $\frac{1}{2m}$ is for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$.

Now we are able to concretely measure the accuracy of our predictor function against the correct results we have so that we can predict new results we don't have.

If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make straight line(defined by $h_\theta(x)$) which passes through this scattered set of data. Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least.

### Gradient Descent

Now that we have the hypothesis function and the cost function which is used to measure how well it fits into data set. We still need a way to estimate the parameters in hypothesis function. So the Gradient Descent comes in.

Given the paramters of our hypothesis function, we can get the value of the cost function. And the goal is to get the smallest value of our cost function. The way we do this is by taking the derivative(the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent, and the size of each step is determined by the parameter $\alpha$, which is called the learning rate.

The gradient descent algorithm is:

repeat until convergence:

$$\theta_j:=\theta_j-\alpha \frac{\partial}{\partial \theta_j} J(\theta_0,\theta_1)$$

where $j=0,1$

#### Gradient Descent for Linear Regression

When specially applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis funtion and modify the equation to:

repeat until convergence: {

$$\theta_0:=\theta_0−\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x_i)-y_i)$$
$$\theta_1:=\theta_1−\alpha\frac{1}{m}\sum_{i=1}^m((h_\theta(x_i)-y_i)x_i)$$
}

where $m$ is the size of the traing set, $\theta_0$ and $\theta_1$ will be changing simultaneously, $x_i$ and $y_i$ are values of the given training set.

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

## Linear Algebra Review

This chapter introduces some basic ideas of linear algebra, including:

1. the difinition of matrix and vector.
2. addition and scalar multiplication.
3. matrix-vector multiplication.
4. matrix-matrix multiplication.
5. matrix multiplication properties, such as not commutative and associative.
6. matrix inverse and transpose.
