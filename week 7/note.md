# Machine Learning Week 7

## Optimization Objective

The **Support Vector Machine(SVM)** is yet another type of *supervised* machine learning algorithm. It is sometimes cleaner and more powerful.

Recall that in logistic regression, we use the following rules:

if y=1, then $h_\theta(x) \approx 1$ and $\theta^T x \ge 0$

if y=0, then $h_\theta(x) \approx 0$ and $\theta^T x \le 0$

Recall the cost function for (unregularized) logistic regression:

$$
h_\theta(x) = \frac{1}{1+e^{-\theta^T x^{(i)}}} \\
J(\theta) = \frac{1}{m}\sum_{i=1}^m-y^{(i)}log(h_\theta(x^{(i)})) - (1-y^{(i)})log(1-h_\theta(x^{(i)}))
$$

To make a support vector machine, we will modify the first term of the cost function $-log(h_\theta(x))$ so that when $\theta^Tx$ is greater than 1, it outputs 0, and for values of z less than 1, we shall use a straight decreasing line instead of the sigmoid curve.

![hinge loss](images/hinge_loss.png)

Similarly, we modify the second term of the cost function $-log(1-h_\theta(x))$ so that when it's less than -1, it outputs 0. We also modify it so that for values of it greater than -1, we use a straight increasing line instead of the sigmoid curve.

![hinge loss 2](images/hinge_loss2.png)

We shall denote these as $cost_1(z)$ and $cost_0(z)$ (respectively, note that $cost_1(z)$ is the cost for classifying when y=1, and $cost_0(z)$ is the cost for classifying when y=0), and we may define them as follows (where k is an arbitrary constant defining the magnitude of the slope of the line):

$$
z = \theta^Tx \\
cost_0(z) = max(0, k(1+z)) \\
cost_1(z) = max(0, k(1-z))
$$

Recall the full cost function from (regularized) logistic regression:

$$
J(\theta) = \frac{1}{m}\sum_{i=1}^m-y^{(i)}log(h_\theta(x^{(i)})) - (1-y^{(i)})log(1-h_\theta(x^{(i)}))+\frac{1}{2m}\sum_{j=1}^n\theta_j^2
$$

We can transform this into the cost function for support vector machines by substituting $cost_0(z)$ and $cost_z(z)$:

$$
J(\theta) = \frac{1}{m}\sum_{i=1}^my^{(i)}cost_1(h_\theta(x^{(i)})) + (1-y^{(i)})cost_0(h_\theta(x^{(i)}))+\frac{1}{2m}\sum_{j=1}^n\theta_j^2
$$

Furthermore, convention dictates that we regularize using a factor C, instead of λ, like so:

$$
J(\theta) = C\sum_{i=1}^m y^{(i)}cost_1(h_\theta(x^{(i)})) + (1-y^{(i)})cost_0(h_\theta(x^{(i)}))+\frac{1}{2}\sum_{j=1}^n\theta_j^2
$$

This is equivalent to multiplying the equation by $C=\frac{1}{\lambda}$, and thus results in the same values when optimized. Now, when we wish to regularize more (that is, reduce overfitting), we decrease C, and when we wish to regularize less (that is, reduce underfitting), we increase C.

Finally, note that the hypothesis of the Support Vector Machine is not interpreted as the probability of y being 1 or 0 (as it is for the hypothesis of logistic regression). Instead, it outputs either 1 or 0. (In technical terms, it is a discriminant function.)

$$
h_\theta(x) =
\begin{cases}
 1 & if \ \theta^Tx \ge 0 \\
 0 & otherwise
\end{cases}
$$

## Large Margin Intuition

A useful way to think about Support Vector Machines is to think of them as *Large Margin Classifiers*.

If y = 1, we want $\theta^T x \ge 1$ (not just ≥0)

If y = 0, we want $\theta^T x \le -1$ (not just <0)

Now when we set our constant C to a very large value (e.g. 100,000), our optimizing function will constrain Θ such that the equation A (the summation of the cost of each example) equals 0. We impose the following constraints on Θ:

$\theta^T x \ge 1$ if y=1 and $\theta^Tx \le -1$ if y=0.

If C is very large, we must choose Θ parameters such that:

$$
\sum_{i=1}^m y^{(i)}cost_1(\theta^Tx) +(1-y^{(i)}) cost_0(\theta^Tx) = 0
$$

This reduces our cost function to $\frac{1}{2}\sum_{j=1}^n\theta_j^2$.

Recall the decision boundary from logistic regression (the line separating the positive and negative examples). In SVMs, the decision boundary has the special property that it is **as far away as possible** from both the positive and the negative examples.

The distance of the decision boundary to the nearest example is called the **margin**. Since SVMs maximize this margin, it is often called a Large Margin Classifier.

The SVM will separate the negative and positive examples by a **large margin**.

This large margin is only achieved when **C is very large**.

Data is **linearly separable** when a **straight line** can separate the positive and negative examples.

If we have **outlier** examples that we don't want to affect the decision boundary, then we can **reduce** C.

Increasing and decreasing C is similar to respectively decreasing and increasing λ, and can simplify our decision boundary.

## Mathmatics Behind Large Margin Classification

Say we have two vectors, u and v:

$$
\vec{u} = (u_1,u_2)\\
\vec{v} = (v_1,v_2)
$$

The length of vector v is denoted ||v|| and it describes the line on a graph from origin (0,0) to $(v_1,v_2)$.

The projection of vector v onto vector u is found by taking a right angle from u to the end of v, creating a right triangle.

- p=length of projection of v onto the vector u
- $u^Tv = p.||u||$

So that:

$$
u^Tv = v^Tu = p.||u||=u_1v_1+u_2v_2
$$

If the angle between the lines for v and u is greater than 90 degrees, then the projection p will be negative.

$$
    min\frac{1}{2}\sum_{j=1}^n\theta_j^2 \\
\begin{aligned}
    & = \frac{1}{2}(\theta_1^2+\theta_2^2+...+\theta_n^2) \\
    &= \frac{1}{2}(\sqrt{\theta_1^2+\theta_2^2+...+\theta_n^2})^2 \\
    &=\frac{1}{2}||\theta||^2
\end{aligned}
$$

We can use the same rules to rewrite $\theta^Tx^{(i)}$:

$$
\theta^Tx^{(i)} = p^{(i)}.||\theta|| = \theta_1x_1^{(i)} + \theta_2x_2^{(i)} + ... + \theta_n x_n^{(i)}
$$

if y=1, we want $p^{(i)}.||\theta||\ge1$

if y=0, we want $p^{(i)}.||\theta||\le-1$

The reason this causes a "large margin" is because: the vector for Θ is perpendicular to the decision boundary. In order for our optimization objective (above) to hold true, we need the absolute value of our projections $p^{(i)}$ to be as large as possible.
