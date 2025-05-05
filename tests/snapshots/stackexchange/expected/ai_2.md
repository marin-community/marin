# Question
Title: What is the significance of weights in a feedforward neural network?
In a feedforward neural network, the inputs are fed directly to the outputs via a series of **weights**.

What purpose do the weights serve, and how are they significant in this neural network?

# Answer
> 5 votes
You described a single-layer feedforward network. They can have multiple layers. The significance of the weights is that they make a linear transformation from the output of the previous layer and hand it to the node they are going to. To say it more simplistically, they specify how important (and in what way: negative or positive) is the activation of node they are coming from to activating the node they are going to.

In your example, since there is only one layer (a row of input nodes and a row of output nodes) it is easy to explain what each node represents. However in multi-layer feedforward networks they can become abstract representations which makes it difficult to explain them and therefore explain what the weights that come to them or go out of them represent.

Another way of thinking about it is that they describe hyperplanes in the space of the output of the previous node layer. If each output from the previous layer represents a point in space, a hyperplane decides which part of the space should give a positive value to the plane's corresponding node in the next layer and which part should give a negative input to it. It actually cuts that space into two halves. If you consider the input space of a multi-layer feedforward network, the weights of the first layer parametrize hyperplanes, however in the next layers they can represent non-linear surfaces in the input space.

---
Tags: neural-networks, weights, perceptron, feedforward-neural-networks
---
