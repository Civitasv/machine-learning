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

Then $h\theta(x)$ will give us the **probability** that our output is 1. For example, $h\theta(x) = 0.7$ gives us the probability of 70% that our output is 1.

$h\theta(x) = P(y=1|x;\theta) = 1-P(y=0|x;\theta)$

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
