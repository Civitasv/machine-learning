# Machine Learning Week 5

## Neural Networks: Learning

### Cost Function

**DEFINE TERMS:**

1. L: total number of layers in the network
2. $s_l$: number of units(not including bias unit) in layer l
3. K: number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote $h_\theta(x)_k$ as being a hypothesis that results in the $k^{(th)}$ output.

Generally, Our cost function for neural networks is going to be a generalization of the one we used for logistic regression.

**The cost function for regularized logistic regression was:**

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$$

**For neural networks, it is going to be slightly more complicated:**

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K[y^{(i)}_klog((h_\theta(x^{(i)}))_k) + (1-y^{(i)}_k)log(1-(h_\theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\theta_{j,i}^{(l)})^2$$

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, between the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer(including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer(excluding the bias unit). As before we did in logistic regression, we square every term.

**Notes:**

- the double sum simply adds up the logistic regression costs calculated for each cell in the output layer; and
- the triple sum simply adds up the squares of all the individual $\theta$ in the entire network.
- the i in the triple sum does not refer to training example i.

### Backpropagation Algorithm

Backpropagation is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression.

Our goal is to compute:

$min_\theta J(\theta)$

That is, we want to minimize our cost function J using an optimal set of parameters in theta.

In this section we'll look at the equations we use to compute the partial derivative of $J(\theta)$:

$\frac{\partial}{\partial\theta_{i,j}^{(l)}} J(\theta)$

In back propagation we're going to compute for every node:

$\delta_j^{(l)}$ : **error** of node j in layer l

Recall that $a_j^{(l)}$ is activation node j in layer l.

For the last layer, we can compute the vector of delta values with:

$\delta^{(L)} = a^{(L)} - y$

Where L is our total number of layers and $a^{(L)}$ is the vector of outputs of the activation units for the last layer. So our **error values** for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y.

And to get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

$\delta^{(l)} = ((\theta^{(l)})^T\delta^{(l+1)}) .* g'(z^{(l)})$

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by $z^{(l)}$.

The g-prime derivative terms can also be written out as:

$g'(u) = g(u) .* (1-g(u))$

Then the full back prpagation equation for the inner nodes is:

$\delta^{(l)} = ((\theta^{(l)})^T\delta^{(l+1)}) .*a^{(l)}.* (1-a^{(l)})$

We can compute our partial derivative terms by multiplying our activation values and our error values for each training example t:

$$\frac{\delta J(\theta)}{\delta\theta_{(i,j)}^{(l)}} = \frac{1}{m}\sum_{t=1}^ma_j^{(t)(l)}\delta_i^{(t)(l+1)}$$

This however ignores regularization, which we'll deal with later.

Now we can take all these equations and put them together into a backpropagation algorithm.

### Put it all together

Given training set ${(x^{(1)}, y^{(1)})...(x^{(m)},y^{(m)})}$

- set $\Delta_{i,j}^{(l)}:=0$ for all $(l,i,j)$

For training example t=1 to m:

- set $a^{(1)} := x^{(t)}$
- Perform forward propagation to compute $a^{(l)}$ for l=2,3,...,L
- Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$
- Compute $\delta^{(L-1)}, \delta^{(L-2)},...,\delta^{2}$ using $\delta^{(l)} = ((\theta^{(l)})^T\delta^{(l+1)}).*a^{(l)}.*(1-a^{(l)})$
- $\Delta_{i,j}^{(l)}:=\Delta_{i,j}^{(l)}+a_j^{(l)}\delta_i^{(l+1)}$ or with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T
- $D_{i,j}^{(l)}:=\frac{1}{m}(\Delta_{i,j}^{(l)} + \lambda\theta_{i,j}^{(l)})$ if $j\not ={0}$
- $D_{i,j}^{(l)} := \frac{1}{m}\Delta_{i,j}^{(l)}$ if $j\not ={0}$

The capital-delta matrix is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative.

The actual proof is quite involved, but, the $D_{i,j}^{(l)}$ terms are the partial derivatives and the results we are looking for:

$$D_{i,j}^{(l)} = \frac{\partial J(\theta)}{\partial\theta_{i,j}^{(l)}}$$

## Backpropagation Intuition

The cost function is:

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K[y^{(i)}_klog((h_\theta(x^{(i)}))_k) + (1-y^{(i)}_k)log(1-(h_\theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\theta_{j,i}^{(l)})^2$$

If we consider simple non-multiclass classification(k=1) and disregard regularization, the cost is computed with:

$$cost(t) = y^{(t)}log(h_\theta(x^{(t)})) + (1-y^{(t)})log(1-h_\theta(x^{(t)}))$$

More intuitively you can think of that equation roughly as:

$cost(t) ≈ (h_\theta(x^{(t)})-y^{(t)})^2$

Intuitively, $\delta_j^{(l)}$ is the error for $a_j^{(l)}$(unit j in layer l)

More formally, the delta values are actually the derivative of the cost function:

$\delta_j^{(l)} = \frac{\partial}{\partial z_j^{(l)}}cost(t)$

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are.

## Implementation Note: Unrolling Parameters

With neural networks, we are working with sets of matrices:

$\theta^{(1)}, \theta^{(2)}, \theta^{(3)},...$

$D^{(1)}, D^{(2)}, D^{(3)},...$

In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

```matlab
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

## Gradient Checking

Gradient checking will assure that our backpropagation works as intended.

We can approximate the derivative of our cost function with:

$\frac{\partial}{\partial \theta}J(\theta) \approx \frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$

With multiple theta matrices, we can approximate the derivative with respect to $\theta_j$ as follows:

$\frac{\partial}{\partial\theta_j}J(\theta) \approx \frac{J(\theta_1,...,\theta_j+\epsilon,...,\theta_n)-J(\theta_1,...,\theta_j-\epsilon,...,\theta_n)}{2\epsilon}$

A good small value for $\epsilon$ guarantees the math above to become true. If the value be much smaller, may we will end up with numerical problems. The professor Andrew usually uses the value $\epsilon = 10^{-4}$.

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

We then want to check that gradApprox ≈ deltaVector.

Once you've verified once that your backpropagation algorithm is correct, then you don't need to compute gradApprox again. The code to compute gradApprox is very slow.

## Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly.

Instead we can randomly initialize our weights:

Initialize each $\theta_{i,j}^{(l)}$ to a random value between $[-\epsilon, \epsilon]$:

$\epsilon = \frac{\sqrt{6}}{L_{output}+L_{input}}$

```matlab
% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

## Steps

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers total.

- Number of input units = dimension of features $x^{(i)}$
- Number of output units = number of classes
- Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
- Defaults: 1 hidden layer. If more than 1 hidden layer, then the same number of units in every hidden layer.

**Training a Neural Network:**

1. Randomly initialize the weights
2. Implement forward propagation to get $h_\theta(x^{(i)})$
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every training example:

```matlab
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

## BP Proof

TODO
