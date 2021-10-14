# Machine Learning Week 2

## Linear Regression with Multiple Variables

Linear Regression with Multiple Variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

1. $x_j^{(i)}$: value of feature j in the $i^{th}$ training example.
2. $x^{(i)}$: the column vector of all the feature inputs of the $i^{th}$ training example.
3. $m$: the number of training examples.
4. $n$: the number of features.

Follow the rules above, we can form the hypothesis function as below:

$$h_\theta(x) = \theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n$$

In order to develop intuition about this function, we can think the $\theta_0$ as the basic, $\theta_1$ as the value per first factor, $\theta_2$ as the value per second factor, ..., $\theta_n$ as the value per $n_{th}$ factor. $x_1$ as the first factor, $x_2$ as the second floor, ..., $x_n$ as the $n_{th}$ floor.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

$$
h_\theta(x) = [\theta_0 \ \theta_1 \ \theta_2 \ ... \theta_n]\left[
 \begin{matrix}
   x_0 \\
   x_1 \\
   x_2
  \end{matrix}
  \right] = \theta^Tx
$$

Note that for convenience reasons in this course Mr.Ng assumes $x_0^{(i)} = 1$ for $(i\in1,...,m)$

The training examples are stored in X row-wise, like such:

$$
X = \left[\begin{matrix}
    x_0^{(1)} \ x_1^{(1)} \\
    x_0^{(2)} \ x_1^{(2)} \\
    x_0^{(3)} \ x_1^{(3)}\end{matrix}
\right],
\theta = \left[ \begin{matrix}
    \theta_0 \\
    \theta_1\end{matrix}\right]
$$

With this definition above, we can calculate the hypothesis as a column vector of size (m\*1) with:

$$h_\theta(X) = X\theta$$

X represents a matrix of training examples $x^{(i)}$ stored row-wise.

### Cost Function

For the paramter vector $\theta$, the cost function is:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$$

We can also form the cost funtion in vector version:

$$J(\theta) = \frac{1}{2m}(X\theta-\vec{y})^T(X\theta-\vec{y})$$

which $\vec{y}$ indicates the vector of all y values.

### Gradient Descent for Multiple Variables

The gradient descent equation itself is generally the same form, we just have to repeat it for our n features:

repeat until convergence:{
$$\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})*x_0^{(i)}$$
$$\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})*x_1^{(i)}$$
$$\theta_2 := \theta_2 - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})*x_2^{(i)}$$
...

}

In other words:

repeat until convergence:{
$$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})*x_j^{(i)}$$

for j:=0..n

}

### Matrix Notation

The Gradient Descent rule can be expressed as:

$$\theta:=\theta-\alpha\nabla J(\theta)$$

where $\nabla J(\theta)$ is a column vector of the form:

$$
\nabla J(\theta) = \left[\begin{matrix}
\frac{\partial J(\theta)}{\partial \theta_0} \frac{\partial J(\theta)}{\partial \theta_1}...\frac{\partial J(\theta)}{\partial \theta_n}
\end{matrix}\right]
$$

And apparentlly the j-th component of the gradient is the summation of the product of two terms:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$

Here, $x_j^{(i)}$, for i=1,..,m, represents the m elements of the j-th column, thus $\vec{x_j}$, of the training set X.

The other term $h_\theta(x^{(i)})-y^{(i)}$ is the vector of the deviations between the predictions $h_\theta(x^{(i)})$ and the true values $y^{(i)}$. Re-writing $\frac{\partial J(\theta)}{\partial \theta_j}$ ,we have:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\vec{x_j}(X\theta-\vec{y})$$

$$\nabla J(\theta) = \frac{1}{m}X^T(X\theta - \vec{y})$$

Finally, the matrix notation of the Gradient Descent rule is:

$$\theta:=\theta-\frac{\alpha}{m}X^T(X\theta-\vec{y})$$

### Feature Normalization

We can speed up gradient descent by having each of our input values in roughly the same range. This is because $\theta$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same range. Ideally:

$$-1\le x_j \le1$$

To be clear, the range don't have to be exactly between -1 and 1, we are only trying to speed things up. The goal is to get all input variables into roughly the same range.

There are two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range of the input variable, resulting a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable, resulting a new average value for the input variable of just 0. To implement both of these techniques, adjust your input values as shown in this formula:

$$x_j := \frac{x_j-\mu_j}{s_j}$$

where $\mu_j$ is the average of all the values for feature j and $s_j$ is the range of values(max-min or standard deviation).

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

### Gradient Descent Tips

**Debugging gradient descent.** Make a plot with number of iterations on the x-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

**Automatic convergence test.** Declare convergence if $J(\theta)$ decreases by less than E in one iteration, where E is some small value such as $10^{-3}$ . However in practice it's difficult to choose this threshold value.

It has been proven that if learning rate $\alpha$ is suffciently small, then $J(\theta)$ will decrease on every iteration.

### Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine $x_1$ and $x_2$ into a new feature $x_3$ by taking $x_1x_2$.

#### Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function(or any other form).

For example, if our hypothesis function is $h_\theta(x) = \theta_0 + \theta_1 x$ the we can create additional features based on $x_1$, to get the quadratic function or the cubic function.

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

eg. if $x_1$ has range 1 - 1000 then range of $x_1^2$ becomes 1 - 1000000 and that of $x_1^3$ becomes 1 - 1000000000.

## Normal Equation

The "Normal Equation" is a method of finding the optimum theta **without iteration**.

$$\theta=(X^TX)^{-1}X^Ty$$

There is **no need** to do feature scaling with the normal equation.

Mathematical proof of the Normal equation requires knowledge of linear algebra and is fairly involved, so you do not need to worry about the details.

Proofs are available at these links for those who are interested:

<https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)>

<http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression>

The following is a comparison of gradient descent and the normal equation:

|      Gradient Descent      |                  Normal Equation                  |
| :------------------------: | :-----------------------------------------------: |
|    Need to choose alpha    |              No need to choose alpha              |
|   Needs many iterations    |                No need to iterate                 |
|         $O(kn^2)$          | $O(n^3)$, cause need to calcute inverse of $X^TX$ |
| Works well when n is large |              Slow if n is very large              |

With the normal equation, computing the inversion has complexity $O(n^3)$, So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

### Normal Equation Noninvertibility

When implementing the normal equation in octave we want to use the 'pinv' function rather than the 'inv'.

$X^TX$ may be **noninvertible**. The common causes are:

- Redundant features, where two features are very closely related
- Too many features(m<=n). In this case, delete some features or use "regularization"(to be explained in a later lesson).

In fact, the deep reason is that there is no solution to the function group.
