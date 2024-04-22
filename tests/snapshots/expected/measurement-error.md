
![](https://statmodeling.stat.columbia.edu/wp-content/uploads/2024/04/Rplot02-copy-copy.jpg)

This is all super-simple; still, it might be useful. In class today a student asked for some intuition as to why, when you’re regressing y on x, measurement error on x biases the coefficient estimate but measurement error on y does not.

I gave the following quick explanation:  

– You’re already starting with the model, y\_i = a + bx\_i + e\_i. If you add measurement error to y, call it y\*\_i = y\_i + eta\_i, and then you regress y\* on x, you can write y\* = a + bx\_i + e\_i + eta\_i, and as long as eta is independent of e, you can just combine them into a single error term.  

– When you have measurement error in x, two things happen to attenuate b—that is, to pull the regression coefficient toward zero. First, if you spreading out x but keep y unchanged, this will reduce the slope of y on x. Second, when you add noise to x you’re changing the ordering of the data, which will reduce the strength of the relationship.

But that’s all words (and some math). It’s simpler and clearer to do a live simulation, which I did right then and there in class!

Here’s the R code:

```
# simulation for measurement error
library("arm")
set.seed(123)
n <- 1000
x <- runif(n, 0, 10)
a <- 0.2
b <- 0.3
sigma <- 0.5
y <- rnorm(n, a + b*x, sigma)
fake <- data.frame(x,y)

fit_1 <- lm(y ~ x, data=fake)
display(fit_1)

sigma_y <- 1
fake$y_star <- rnorm(n, fake$y, sigma_y)
sigma_x <- 4
fake$x_star <- rnorm(n, fake$x, sigma_x)

fit_2 <- lm(y_star ~ x, data=fake)
display(fit_2)

fit_3 <- lm(y ~ x_star, data=fake)
display(fit_3)

fit_4 <- lm(y_star ~ x_star, data=fake)
display(fit_4)

x_range <- range(fake$x, fake$x_star)
y_range <- range(fake$y, fake$y_star)

par(mfrow=c(2,2), mar=c(3,3,1,1), mgp=c(1.5,.5,0), tck=-.01)
plot(fake$x, fake$y, xlim=x_range, ylim=y_range, bty="l", pch=20, cex=.5)
abline(coef(fit_1), col="red", main="No measurement error")
plot(fake$x, fake$y_star, xlim=x_range, ylim=y_range, bty="l", pch=20, cex=.5)
abline(coef(fit_2), col="red", main="Measurement error on y")
plot(fake$x_star, fake$y, xlim=x_range, ylim=y_range, bty="l", pch=20, cex=.5)
abline(coef(fit_3), col="red", main="Measurement error on x")
plot(fake$x_star, fake$y_star, xlim=x_range, ylim=y_range, bty="l", pch=20, cex=.5)
abline(coef(fit_4), col="red", main="Measurement error on x and y")

```

The resulting plot is at the top of this post.

I like this simulation for three reasons:

1. You can look at the graph and see how the slope changes with measurement error in x but not in y.

2. This exercise shows the benefits of clear graphics, including little things like making the dots small, adding the regression lines in red, labeling the individual plots, and using a common axis range for all four graphs.

3. It was fast! I did it live in class, and this is an example of how students, or anyone, can answer this sort of statistical question directly, with a lot more confidence and understanding than would come from a textbook and some formulas.

**P.S.** As Eric Loken and I discuss in [this 2017 article](http://www.stat.columbia.edu/~gelman/research/published/measurement.pdf), everything gets more complicated if you condition on "statistical significance."

**P.P.S.** Yes, I know my R code is ugly. Think of this as an inspiration: even if, like me, you’re a sloppy coder, you can still code up these examples for teaching and learning.

