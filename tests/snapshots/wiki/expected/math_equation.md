# Method of undetermined coefficients - Wikipedia

In mathematics, the **method of undetermined coefficients** is an approach to finding a particular solution to certain nonhomogeneous ordinary differential equations and recurrence relations. It is closely related to the annihilator method, but instead of using a particular kind of differential operator (the annihilator) in order to find the best possible form of the particular solution, an ansatz or 'guess' is made as to the appropriate form, which is then tested by differentiating the resulting equation. For complex equations, the annihilator method or variation of parameters is less time-consuming to perform. 

Undetermined coefficients is not as general a method as variation of parameters, since it only works for differential equations that follow certain forms. 

## Description of the method

Consider a linear non-homogeneous ordinary differential equation of the form 

where  denotes the i-th derivative of , and  denotes a function of . The method of undetermined coefficients provides a straightforward method of obtaining the solution to this ODE when two criteria are met: 

1. are constants.
2. *g*(*x*) is a constant, a polynomial function, exponential function , sine or cosine functions  or , or finite sums and products of these functions (,  constants).

The method consists of finding the general homogeneous solution  for the complementary linear homogeneous differential equation 

and a particular integral  of the linear non-homogeneous ordinary differential equation based on . Then the general solution  to the linear non-homogeneous ordinary differential equation would be 

If  consists of the sum of two functions  and we say that  is the solution based on  and  the solution based on . Then, using a superposition principle, we can say that the particular integral  is 

## Typical forms of the particular integral

In order to find the particular integral, we need to 'guess' its form, with some coefficients left as variables to be solved for. This takes the form of the first derivative of the complementary function. Below is a table of some typical functions and the solution to guess for them. 

| Function of *x* | Form for *y* |
| --- | --- |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

If a term in the above particular integral for *y* appears in the homogeneous solution, it is necessary to multiply by a sufficiently large power of *x* in order to make the solution independent. If the function of *x* is a sum of terms in the above table, the particular integral can be guessed using a sum of the corresponding terms for *y*. 

## Examples

### Example 1

Find a particular integral of the equation 

The right side *t* cos *t* has the form 

with *n* = 2, *α* = 0, and *β* = 1. 

Since *α* \+  = *i* is *a simple root* of the characteristic equation 

we should try a particular integral of the form 

Substituting *y*<sub>*p*</sub> into the differential equation, we have the identity 

Comparing both sides, we have 

which has the solution 

We then have a particular integral 

### Example 2

Consider the following linear nonhomogeneous differential equation: 

This is like the first example above, except that the nonhomogeneous part () is *not* linearly independent to the general solution of the homogeneous part (); as a result, we have to multiply our guess by a sufficiently large power of *x* to make it linearly independent. 

Here our guess becomes: 

By substituting this function and its derivative into the differential equation, one can solve for *A*: 

So, the general solution to this differential equation is: 

### Example 3

Find the general solution of the equation: 

is a polynomial of degree 2, so we look for a solution using the same form, 

Plugging this particular function into the original equation yields, 

which gives: 

Solving for constants we get: 

To solve for the general solution, 

where  is the homogeneous solution , therefore, the general solution is: