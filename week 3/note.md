# Machine Learning Week 3

## Classification and Representation

Now we are switching from regression problems to **classification problems.** Don't be confused by the name "Logistic Regression"; it is named that way for historical reasons and is actually an approach to classification problems, not regression problems.

### Binary Classification

Instead of our output vector y being a continuous range of values, it will only be 0 or 1.

$y\in\{0,1\}$

Where 0 is usually taken as the "negative class" and 1 as the "positive class", but you are free to assign any representation to it.

One method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. This method doesn't work well because classification is not actually a linear function.

Our hypothesis should satisfy:

$0\le h_\theta(x) \le 1$

Our new form uses the "Sigmoid Function", also called the "Logistic Function":

$h_\theta(x) = g(\theta^Tx)$

$z = \theta$

$g(z) = \frac{1}{1+e^{-z}}$

![logistic function](images/logistic_function.png)

The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

Then $h_\theta(x)$ will give us the **probability** that our output is 1. For example, $h_\theta(x) = 0.7$ gives us the probability of 70% that our output is 1.

$h_\theta(x) = P(y=1|x;\theta) = 1-P(y=0|x;\theta)$

$P(y=0|x;\theta)+P(y=1|x;\theta) = 1$

### Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

$h_\theta(x) \ge 0.5 \rightarrow y=1$

$h_\theta(x) < 0.5 \rightarrow y=0$

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

$g(z) \ge 0.5 \\ when \ z \ge 0$

So if pur input to g is $\theta^TX$, then that means:

$h_\theta(x) = g(\theta^Tx)\ge0.5 \\
when \ \theta^Tx \ge 0$

From these statements we can now say:

$\theta^Tx \ge 0 \rightarrow y=1 \\
\theta^Tx < 0 \rightarrow y=0$

The decision boundary is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

## Logistic Regression Model

### Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:

$J(\theta) = \frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)},y^{(i)})$

$Cost(h_\theta(x),y) = -log(h_\theta(x)) \ if \ y=1\\
Cost(h_\theta(x),y) = -log(1-h_\theta(x)) \ if \ y=0$

The more our hypothesis is off from y, the larger the cost function output. If our hypothesis is equal to y, then our cost is 0:

$Cost(h_\theta(x),y) = 0 \ if \ h_\theta(x) = y \\
Cost(h_\theta(x),y) \rightarrow \infty \ if \ y = 0 \ and \ h_\theta(x) \rightarrow 1 \\
Cost(h_\theta(x),y) \rightarrow \infty \ if \ y = 1 \ and \ h_\theta(x) \rightarrow 0$

Note that writing the cost function in this way guarantees that $J(\theta)$ is convex for logistic regression.

### Simplified Cost Function and Gradient Descent

We can compress our cost function's two conditional cases into one case:

$Cost(h_\theta(x),y) = -ylog(h_\theta(x)) - (1-y)log(1-h_\theta(x))$

So that our cost function will be:

$J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))]$

A vectorized implementation is:

$h = g(X\theta) \\
J(\theta) = \frac{1}{m}(-y^Tlog(h) - (1-y)^Tlog(1-h))$

#### Gradient Descent

Remember that the general form of gradient descent is:

$Repeat \{ \\
\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta)\\
\}$

We can work out the derivative part using calculus to get:

$Repeat \{ \\
\theta_j := \theta_j - \frac {\alpha}{m} \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}\\
\}$

Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

$\theta:=\theta-\frac{\alpha}{m}X^T(g(X\theta)-\vec{y})$

### Advanced Optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. A. Ng suggests not to write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave and matlab both provide them.

To use advanced optimization, we first need to provide a function that evaluates the following two functions for a given input value $\theta$:

$J(\theta) \\
\frac{\partial}{\partial\theta_j}J(\theta)$

We can write a single function that returns both of these:

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use the **fminunc()** optimization algorithm along with the **optimset()** function that create an object containing the options we want to send to **fminunc**.

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
      initialTheta = zeros(2,1);
      [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

### Multiclass Classification: One-vs-all

Now we will approach the classification of data into more than two categories. Instead of y={0,1}, we will expand our definition so that y={0,1,...,n}.

In this case we divide our problem into n+1 binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

$y\in {0,1,...,n} \\
h_\theta^{(0)} = P(y=0|x;\theta) \\
h_\theta^{(1)} = P(y=1|x;\theta) \\
... \\
h_\theta^{(n)} = P(y=n|x;\theta) \\
prediction = max_i(h_\theta^{(i)}(x))$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

## Regularization

**The Problem of Overfitting:** High bias or underfitting is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. eg. if we take $h_\theta(x) = \theta_0+\theta_1x_1+\theta_2x_2$ then we are making an initial assumption that a linear model will fit the training data well and will be able to generalize but actually that may not be the case.

At the other extreme, overfitting or high variance is caused by a hypothesis function that fits the avaliable data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1. Reduce the number of features:

   a) Manually select which features to keep.

   b) Use a model selection algorithm(studied later in the course).

2. Regularization

Keep all the features, but reduce the parameters $\theta_j$.

Regularization works well when we have a lot of slightly useful features.

### Regularized Linear Regression

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

#### Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms on our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

$\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4$

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:

$min_\theta \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 + 1000\theta^3+1000\theta^4$

We've added two extra terms at the end to inflate the cost of $\theta_3$ and $\theta_4$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_3$ and $\theta_4$ to near zero.

Using this method, we could also regularize all of our theta parameters in a single summation:

$min_\theta\frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^n\theta_j^2]$

The λ, or lambda, is the regularization parameter. It determines how much the costs of our theta parameters are inflated. You can visualize the effect of regularization in this interactive plot : <https://www.desmos.com/calculator/1hexc8ntqp>

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting.

#### Gradient Descent

We will modify our gradient function to separate out $\theta_0$ from the rest of the parameters because we do not want to penalize $\theta_0$.

$Repeat \{ \\
\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} \\
\theta_j := \theta_j - \alpha[(\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)})+\frac{\lambda}{m}\theta_j] \ \ j \in \{1,2,...,n\} \\
\}$

The term $\frac{\lambda}{m}\theta_j$ performs our regularization.

With some manipulation our update rule can also be represented as:

$\theta_j := \theta_j(1-\alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

The first term in the above equation, $1-\alpha\frac{\lambda}{m}$ will always be less than 1. Intuitively you can see it as reducing the value of $\theta_j$ by some amount on every update.

Notice that the second term is now exactly the same as it was before.

### Normal Equation

Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

$\theta = (X^TX+\lambda L)^{-1}X^Ty \\
where \ L =\left [ \begin{matrix}
   0 &  &&  &&\\
    & 1 &&&&  \\
    &  & 1&&& \\
    &&&.&& \\
    &&&&.& \\
    &&&&&1 \\
  \end{matrix} \right]$

L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including $x_0$), multiplied with a single real number λ.

Recall that if $m \le n$, then $X^TX$ is non-invertible. However, when we add the term $\lambda L$, then $X^TX+\lambda L$ becomes invertible.

### Regularized Logistic Regression

We can regularized logistic regression in a similar way that we regularize linear regression. Let's start with the cost function.

#### Cost Function

Recall that our cost function for logistic regression was:

$J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))]$

We can regularized this equation by adding a term to the end:

$J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$

#### Gradient Descent

Just like with linear regression, we will want to separately update $\theta_0$ and the rest of the parameters because we do not want to regularize $\theta_0$.

$Repeat \{ \\
\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} \\
\theta_j := \theta_j - \alpha[(\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)})+\frac{\lambda}{m}\theta_j] \ \ j \in \{1,2,...,n\} \\
\}$

This is identical to the gradient descent function presented for linear regression.
