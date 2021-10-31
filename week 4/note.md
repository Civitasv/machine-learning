# Machine Learning Week 4

## Neural Networks: Representation

### Non-linear Hypotheses

Performing linear regression with a complex set of data with many features is very unwieldy. Say you wanted to create a hypothesis from three features that included all the quadratic terms:

$g(\theta_0+\theta_1x_1^2+\theta_2x_1x_2+\theta_3x_1x_3+\theta_4x_2^2+\theta_5x_2x_3+\theta_6x_3^2)$

That gives us 6 features. And for 100 features, if we want to make them quadratic we would get 5050 resulting new features.

We can approximate the growth of the number of new features we get with all quadratic terms with $O(\frac{n^2}{2})$. The features would grow asymptotically at $O(n^3)$ if we want to include all the cubic terms. These are very steep growths, so as the number of our features increase, the number of quadratic or cubic features increase very rapidly and becomes quickly impractical.

Example: let our training set be a collection of 50 x 50 pixel black-and-white photographs, and our goal will be to classify which ones are photos of cars. Our feature set size is then n = 2500 if we compare every pair of pixels.

Now let's say we need to make a quadratic hypothesis function. With quadratic features, our total features will be about $2500^2 / 2 = 3125000$.which is very impractical.

Thus, Neural networks offers an alternate way to perform machine leanring when we have complex hypotheses with many features.

### Neurons and the Brain

Neural networks are limited imitations of how our own brains work. They've had a big recent resurgence because of advances in computer hardware.

There is evidence that the brain uses only one "learning algorithm" for all its different functions. Scientists have tried cutting (in an animal brain) the connection between the ears and the auditory cortex and rewiring the optical nerve with the auditory cortex to find that the auditory cortex literally learns to see.

This principle is called "neuroplasticity" and has many examples and experimental evidence.

### Model Representation I

At a very simple level, neurons are basically computational units that take input as electrical input that are channeled to outputs.

In our model, our inputs are like the input features $x_1,x_2,...,x_n$, and the output is the result of our hypothesis function.

Attention that our $x_0$ input node is called "bias unit" and it is always equal to 1.

In neural networks, we use the same logistic function as in classification: $\frac{1}{1+e^{-\theta^Tx}}$. In neural networks however we sometimes call it a sigmoid activation function.

Our "theta" parameters are sometimes instead called "weights" in the neural networks model.

Visually, a simplistic representation looks like:

![neuron model](images/neuron%20model.png)

Our input nodes (layer 1) go into another node (layer 2), and are output as the hypothesis function.

The first layer is called the "input layer" and the final layer is called the "output layer", which gives the final value computed on the hypothesis.

We can have intermediate layers of nodes between the input and output layers called the "hidden layer".

We label these intermediate or hidden later nodes $a_0^2,..,a_n^2$ and call them "activation units".

$a_i^{(j)}$: "activation" of unit i in layer j

$\theta^{(j)}$: matrix of weights controlling function mapping from layer j to layer j+1

If we had one hidden layer, it would look visually something like:

![one hidden layer](images/one%20hidden%20layer.png)

The values for each of the "activation" nodes is obtained as follows:

$a_1^{(2)} = g(\theta_{10}^{(1)}x_0 + \theta_{11}^{(1)}x_1 + \theta_{12}^{(1)}x_2 + \theta_{13}^{(1)}x_3)$

$a_2^{(2)} = g(\theta_{20}^{(1)}x_0 + \theta_{21}^{(1)}x_1 + \theta_{22}^{(1)}x_2 + \theta_{23}^{(1)}x_3)$

$a_3^{(2)} = g(\theta_{30}^{(1)}x_0 + \theta_{31}^{(1)}x_1 + \theta_{32}^{(1)}x_2 + \theta_{33}^{(1)}x_3)$

$h_\theta(x) = a_1^{(3)} = g(\theta_{10}^{(2)}a_0^{(2)} + \theta_{11}^{(2)}a_1^{(2)} + \theta_{12}^{(2)}a_2^{(2)} + \theta_{13}^{(2)}a_3^{(2)})$

The dimension of the $\theta^{(j)}$ is determined as follows:

if network has $s_j$ units in layer j and $s_{j+1}$ units in layer j+1, then $\theta_{(j)}$ will be of dimension $s_{j+1} \times (s_j+1)$.

The +1 comes from the addition in $\theta_{(j)}$ of the bias nodes, $x_0$ and $\theta_0^{(j)}$. In other words the output nodes will not include the bias nodes while the inputs will.

### Model Representation II

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $z_k^{(j)}$ that encompasses the parameters inside our g function.

$a_1^{(2)} = g(z_1^{(2)})$

$a_2^{(2)} = g(z_2^{(2)})$

$a_3^{(2)} = g(z_3^{(2)})$

In other words, for layer j=2 and node k, the variable z will be:

$z_k^{(2)} = \theta_{k,0}^{(1)}x_0 + \theta_{k,1}^{(1)}x_1 + ... + \theta_{k,n}^{(1)}x_n$

The vector representation of x and $z^j$ is:

$x = \left [ \begin{matrix}
   x_0 \\
   x_1 \\
   ... \\
   x_n
\end{matrix} \right ]$
$z^{(j)} = \left [ \begin{matrix}
   z_1^{(j)} \\
   z_2^{(j)} \\
   ... \\
   z_n^{(j)}
\end{matrix} \right ]$

Setting $x = a^{(1)}$, we can rewrite the equation as:

$z^{(j)} = \theta^{(j-1)}a^{(j-1)}$

We can the add a bias unit to layer j after we have computed $a^{(j)}$. This will be element $a_0^{(j)}$ and will be equal to 1.

To compute our final hypothesis, let's first compute another z vector:

$z^{(j+1)} = \theta^{(j)}a^{(j)}$

We then get our final result with:

$h_\theta(x) = a^{(j+1)} = g(z^{(j+1)}$

Notice that in this **last step**, between layer j and layer j+1, we are doing **exactly the same thing** as we did in logistic regression.

Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

## Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four final resulting classes:

$\left [ \begin{matrix}
   x_0 \\
   x_1 \\
   ... \\
   x_n
\end{matrix} \right ]$
$\rightarrow$
$\left [ \begin{matrix}
   a_0^{(2)} \\
   a_1^{(2)} \\
   ... \\
   a_n^{(2)}
\end{matrix} \right ]$
$\rightarrow$
$\left [ \begin{matrix}
   a_0^{(3)} \\
   a_1^{(3)} \\
   ... \\
   a_n^{(3)}
\end{matrix} \right ]$
$\rightarrow$
$\left [ \begin{matrix}
   h_\theta(x)_1 \\
   h_\theta(x)_2\\
   ... \\
   h_\theta(x)_n
\end{matrix} \right ]$

Our final layer of nodes, when multiplied by its theta matrix, will result in another vector, on which we will apply the g() logistic function to get a vector of hypothesis values.

Our resulting hypothesis for one set of inputs may look like:

$h_\theta(x) = \left [ \begin{matrix}
   0 \\
   0 \\
   1 \\
   0
\end{matrix} \right ]$

In which case our resulting class is the third one down.

We can define our set of resulting classes as y:

$y^{(i)} \in \left [ \begin{matrix}
   1 \\
   0 \\
   0 \\
   0
\end{matrix} \right ]$,
$\left [ \begin{matrix}
   0 \\
   1 \\
   0 \\
   0
\end{matrix} \right ]$,
$\left [ \begin{matrix}
   0 \\
   0 \\
   1 \\
   0
\end{matrix} \right ]$,
$\left [ \begin{matrix}
   0 \\
   0 \\
   0 \\
   1
\end{matrix} \right ]$

Our final value of our hypothesis for a set of inputs will be one of the elements in y.
