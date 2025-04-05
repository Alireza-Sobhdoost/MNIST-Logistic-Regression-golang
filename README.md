# Introduction

This report presents the implementation of a logistic regression model
using the Go programming language. The model is trained on a binary
classification task using the MNIST dataset (two classes). The primary
goal of the project is to classify handwritten digits efficiently using
a gradient descent algorithm with Adam optimization.

# Logistic Regression Model

Logistic regression is a linear model that estimates the probability of
a binary outcome using the sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$ where $z = w^T x + b$.\
The loss function used for logistic regression is the binary
cross-entropy loss:
$$L(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

# Adam Optimization

Adam (Adaptive Moment Estimation) combines the advantages of two popular
optimization algorithms: AdaGrad and RMSProp. It computes adaptive
learning rates for each parameter using first and second moments of the
gradients. The Adam optimizer has the following key hyperparameters:

-   $\alpha$ (learning rate): Controls the step size during gradient
    descent. we set to 0.001.

-   $\beta_1$ (exponential decay rate for the first moment estimates):
    we set to 0.9.

-   $\beta_2$ (exponential decay rate for the second moment estimates):
    we set to 0.999.

-   $\epsilon$ (small constant to prevent division by zero): we set to
    $10^{-8}$.

The update equations for Adam are as follows:

## Explanation of $m_t$ (First Moment Estimate)

In the context of the Adam optimizer, $m_t$ represents the **first
moment estimate** of the gradients. It is the moving average of the
gradients, which helps in smoothing out fluctuations in the gradient and
providing a more stable update.

The first moment estimate captures the **gradient values** themselves
and helps in adjusting the learning rate based on the average of the
gradients over time. The update equation for $m_t$ is as follows:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$ Where:

-   $g_t$ is the gradient at time step $t$,

-   $\beta_1$ is the exponential decay rate for the first moment
    estimate (we set to 0.9).

The first moment estimate helps to reduce the impact of noisy gradients
by averaging them over time. However, like $v_t$, the first moment
estimate is **biased towards zero** in the initial steps, especially
when $t$ is small. To correct this bias, Adam uses a **biased-corrected
first moment estimate**, which is given by:
$$m_t^\text{corrected} = \frac{m_t}{1 - \beta_1^t}$$

## Explanation of $v_t$ (Second Moment Estimate)

In the context of the Adam optimizer, $v_t$ represents the **second
moment estimate** of the gradients. It is the moving average of the
squared gradients, which helps in adjusting the learning rate based on
the variance (spread) of the gradients.

The second moment estimate captures the **squared gradients** and
provides information about the variance of the gradients over time. The
update equation for $v_t$ is as follows:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$ Where:

-   $g_t$ is the gradient at time step $t$,

-   $\beta_2$ is the exponential decay rate for the second moment
    estimate (usually set to 0.999).

The second moment estimate helps in preventing large updates during
training, especially when the gradients are large or noisy.

However, the second moment estimate $v_t$ is **biased towards zero**
during the initial training steps, especially when $t$ is small. To
correct this bias, the Adam optimizer uses the **biased-corrected second
moment estimate**, which is given by:
$$v_t^\text{corrected} = \frac{v_t}{1 - \beta_2^t}$$

## Updates

This correction makes the second moment estimate unbiased, especially in
the early stages of training.

Once the moment estimates ($m_t$ and $v_t$) are computed, the parameter
update rule for weights are as follows (The bias is updated similarly):
$$w = w - \alpha \frac{m_t^\text{corrected}}{\sqrt{v_t^\text{corrected}} + \epsilon}$$
Where:

-   $m_t^\text{corrected}$ is the corrected first moment estimate,

-   $v_t^\text{corrected}$ is the corrected second moment estimate,

-   $\alpha$ is the learning rate,

-   $\epsilon$ is a small constant to prevent division by zero.

This update helps to scale the learning rate for each parameter based on
both the magnitude and variance of the gradients, making training more
stable and efficient.

# Image Processing and Data Handling

The MNIST dataset consists of 28x28 grayscale images. In this
implementation, we are able to choose different classes from the
dataset. For demonstration purposes, digits 0 and 1 are selected. The
dataset is split into training and testing sets with a ratio of 0.7.
Images are loaded, resized, and normalized before being fed into the
model. To normalize the pixel values to a range of \[0, 1\], we divide
each pixel value by 255, effectively applying a semi-min-max
normalization technique. Each image is then flattened into a
784-dimensional vector. The labels are converted to binary classes
(e.g., digit 0 and digit 1).

# Training Procedure

The model is trained via 1000 epoch and using batches *(size = 2)* of
data to improve computational efficiency. The gradient of the loss
function with respect to the model parameters is computed to update the
weights and biases. During the optimization process, the gradient and
parameter updates are efficiently calculated using the Gonum library,
which provides high-performance numerical operations in Go. This enables
fast matrix operations and vectorized calculations essential for
large-scale data processing. The loss is calculated using binary
cross-entropy, and gradients are computed with respect to the model
parameters. The Adam optimizer then updates the weights and biases.

# Performance Metrics

The model's performance is evaluated using precision, recall, and
F1-score metrics:

-   Precision: The proportion of correctly predicted positive
    observations to the total predicted positives.

-   Recall: The proportion of correctly predicted positive observations
    to the all actual positives.

-   F1-score: The weighted average of precision and recall, balancing
    the two when they are in conflict.

The results obtained for this model are as follows:\

    Metric     Value
  ----------- --------
   Precision   0.9965
    Recall       1
   F1-score    0.9982

  : Performance metrics of the logistic regression model on the MNIST
  dataset.

# Model Saving and Metrics Calculation

After training, the model is saved in JSON format, which includes
weights and biases. Performance metrics such as precision, recall, and
F1-score are calculated to evaluate the model.\
The loading code is also avalable to test if the result are real or
not.\
The model is loaded from the JSON file, and predictions are made on the
test set. The performance metrics are calculated again to ensure
consistency.

# Conclusion

This report demonstrated the implementation of logistic regression using
Go, focusing on gradient descent optimization with Adam. The model
effectively classifies digits from the MNIST dataset, showcasing the
power of the Adam optimizer in stabilizing and accelerating convergence.
