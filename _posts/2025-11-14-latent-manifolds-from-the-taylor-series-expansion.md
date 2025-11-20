---
layout: post
title: Latent Manifolds From the Taylor Series Expansion (IN PROGRESS)
date: 2025-11-14 10:14:00-0400
description: TODO figures, simpler wording, less mathematically overwhelming, proofreading # Derivation of polynomial manifolds / spectral submanifolds from the Taylor series expansion, with additional discourse on properties of the corresponding encoders and decoders.
tags: ML optimization dimensionality-reduction
categories: numerical-methods
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

## Latent Variables of a Dynamical System

Computational physics simulations often solve large systems of coupled ordinary differential equations (ODEs). Such systems may naturally model the physics (e.g., molecular dynamics); other times, they may result from a method of lines discretization of an unsteady state partial differential equation (PDE)---discretization in space but not in time. Regardless of their origin, these systems abound and can often be modeled as a deceptively simple equation:

$$
\begin{equation}
    \frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t),
\end{equation}
$$

where $\mathbf{y}$ is the state of the dynamical system, $t$ is time, and $f$ is the vector field of the dynamical system, which models how the state evolves over time.

In this dynamical system, $\mathbf{y} = \mathbf{y}(t)\in\mathbb{R}^{N}$ since the state has $N$ degrees of freedom. As an example, let's say an engineer wants to model the diffusion of temperature through a 12-inch metal rod after a candle is placed near it on the left end. In real--life there are an uncountably infinite number of positions along the rod: $1.1$in from the left, $1.11$in from the left, $1.111$in from the left, and so on ad infinitum. This means that one can analytically model the system's temperature as a function of space and time, yielding the following PDE:

$$
\begin{equation}
    \frac{\partial T}{\partial t} = \nu \frac{\partial^2}{\partial x^2} T,
\end{equation}
$$

where $\nu$ is a diffusion constant quantifying the thermal conductivity of the rod, $\frac{\partial^2}{\partial x^2}$ is a 2nd--order partial differential operator in space, and $T\in\mathbb{R}^{\infty}$ is temperature modeled as a function of location $x$ and time $t$. The algebraic structure of this equation and the spectral properties of the differential operators encode the physics.

Computers, however, have limited memory, so the aforementioned infinite continuum of positions along the 1D rod must be discretized into a finite collection of cells spanning the 12 inches of metal. The number of discretization cells is effectively the spatial resolution of the computational physics model. Using the method of lines to semi--discretize the 1D linear heat equation yields the following system of ODEs:

$$
\begin{equation}
    \frac{d\mathbf{y}}{dt} = \nu \mathbf{A} \mathbf{y},
\end{equation}
$$

where $\mathbf{y}\in\mathbb{R}^N$ with $N$ being the number of discretization cells.

The integer, $N$, is arbitrary, but larger values correspond to higher spatial resolution, which may be necessary for resolving details. Unfortunately, increasing $N$ increases computational costs in both memory and time since more data must be stored and processed. This is just like how having a 4K resolution image of me is great since you have $N = 3840 \times 2160 = 8,294,400$ pixels to resolve fine details with but such a 4K image requires much more storage than a Full HD image with $N = 1920 \times 1080 = 2,073,600$ pixels. 

The number $N$ is arbitrary---you can still simulate diffusion with $N=10$, $N=100$, or $N=1,000,000$! Importantly though, the actual physics are aptly modeled with a small number of latent variables that we don't directly observe. In other words, even though we have $N$ physical observables (i.e., the temperature readings at each of our $N$ cells), there are actually $n\ll N$ latent variables that evolve over time! This is best understood geometrically: the state of a dynamical system can be envisioned as a point in $\mathbb{R}^N$ that flies around over time; in physics, the path traced out happens to be some curve in low--dimensional surface. To better imagine this concept, put your index finger on your nose then move your finger back and forth as if you were tracing out Pinocchio's nose. This curve you traced out with your finger is a 1--dimensional curve embedded in a 3--dimensional space, so positions along the curve can aptly be described with a single latent variable, that being the "distance from nose" instead of 3 different physical observables $x$, $y$, and $z$! In physics, the same logic applies, but we can take computer simulations with hundreds of millions of physical observables but simulate them in the latent manifolds rather than in the high--dimensional physical observable space, yielding astronomical speedups. As an example, researchers have taken a soft robot with $N\approx 800,000$ degrees of freedom but controlled it by actuating $n=2$ latent variables.

The identification and use of latent manifolds from data is a field of machine learning called dimensionality reduction; it is paramount to model order reduction in computational physics, principal component analysis in statistics, and autoencoder architectures in deep learning--based artificial intelligence. This post discusses how to infer latent manifolds from snapshots of the state of a physical system under the assumption that the manifold is well--modeled by a Taylor series expansion. In doing so, we will derive the data--driven optimization problem used to infer what some researchers in model order reduction call a polynomial manifold but others call a spectral submanifold.


## The Taylor Series Expansion

Recall that our goal is to sample the state of a dynamical system's physical observables over time and use these data to parameterize a "surface" upon which our data live. The word surface is used loosely here since it can be 1--dimensional (a curve), 2--dimensional (a fabric), or of any higher dimension. These surfaces are called manifolds and possess important mathematical structure that is the subject of immense study in mathematics through differential geometry, topology, and more. Regardless, it helps to make assumptions on our inferred parameterization of a latent manifold, such as its dimensionality and nonlinearity. Assuming that our latent manifold is a function of the corresponding system's latent variables, we can impose different functions as structures to represent it. In this post we'll assume the parameterization as a power series function of latent variables; more specifically, as a Taylor series function of latent variables.

Taylor's Theorem states that an $m$ times continuously differentiable function can be approximated by its $n$th degree Taylor polynomial plus a remainder. This post assumes (for simplicity) that the latent manifold under study is an analytic function such that it is infinitely differentiable and possesses a convergent Taylor series expansion about every point, but this is not true for all dynamical systems since not all vector fields are analytic. Researchers in the spectral submanifold community strongly emphasize the importance of the continuous differentiability of the Taylor series latent manifold but others those who call these latent manifolds polynomial manifolds tend to not do so.

Let's first define the Taylor series expansion of a multivariate, vector--valued function, $\mathbf{y}:\mathbb{R}^{n}\rightarrow\mathbb{R}^N$, where $(\cdot)^{\otimes p}:\mathbb{R}^N\rightarrow\mathbb{R}^{n^p}$ denotes the $p$th Kronecker power and $[\mathrm{D}^p\mathbf{y}]_{\mathbf{x}_0}$ denotes the tensor of all $p$th--order partial derivatives of elements of $\mathbf{y}$ with respect to elements of $\mathbf{x}$---each evaluated at $\mathbf{x}_0$ after derivative computation:

$$
\begin{equation}
    \mathbf y(\mathbf x) = \mathbf y(\mathbf x_0) + \left[\mathrm{D}\mathbf y\right]_{\mathbf x_0}(\mathbf x - \mathbf x_0) + \frac{1}{2}\left[\mathrm{D}^2\mathbf y\right]_{\mathbf x_0}(\mathbf x - \mathbf x_0)^{\otimes2} + \cdots.
\end{equation}
$$

The rest of this post will consider the Maclaurin expansion (i.e., $\mathbf{x}_0\equiv\mathbf{0}$):

$$
\begin{equation}\label{eqn:maclaurin_series}
    \mathbf y(\mathbf x) = \mathbf f(0) + \left[\mathrm{D}\mathbf y\right]_{0}\,\mathbf x + \frac{1}{2}\left[\mathrm{D}^2\mathbf y\right]_{0}\,\mathbf x^{\otimes2} + \cdots.
\end{equation}
$$

Expanding our Taylor series about the origin ensures that we can construct a non--affine subspace to serve as the abscissae of our inferred latent manifold. In particular, we will build the latent manifold as a graph over this subspace spanned by the latent variables. As we will later show, the range of the linear decoder, $\mathbf{V}_1$, can well--approximate the latent manifold's tangent space about the origin since it is inferred from data as the aforementioned Taylor expansion's Jacobian.

Leveraging a Maclaurin expansion does not, however, mean that the physical observables, $\mathbf{y}_i\in\mathbb{R}^N$ naturally exist about the origin. In practice, these snapshots of the dynamical system's state are shifted about the origin, but various methods exist to do so. In polynomial manifold literature, the mean snapshot is typically chosen as the anchor of the abscissa but the initial snapshot has also been used. However, in spectral submanifold literature, a hyperbolic fixed point of the dynamical system is shifted to the origin to anchor the abscissa. Those familiar with linear principal component analysis (PCA) / proper orthogonal decomposition (POD) likely recognize this step of shifting data about the origin (usually using the mean datum as the origin). This is because, as we'll see, linear PCA/POD methods construct latent manifolds as subspaces. They do this by truncating the multivariate, vector--valued Maclaurin series expansion to neglect quadratic and higher--order nonlinear structures in their data.

The following subsections derive Taylor expansions for the different values of $n$ and $N$. Pardon me for using both "Taylor expansion" and "Maclaurin expansion" terminology in this post. Both terminologies are correct anyway since a Maclaurin expansion is a special case of the Taylor expansion and we will assume that our latent manifolds are constructed about the origin.


### One Input, One Output $(n=1,N=1)$

In this subsection we consider the Taylor expansion of a univariate, scalar--valued function, where $\mathbf{y}\equiv y$, $\mathbf{x}\equiv x$, and $\mathbf{0}\equiv 0$:

$$
\begin{equation}
\begin{aligned}
    \underbrace{ y(x) }_{1\times 1}
    &= \underbrace{ y(0) }_{1\times 1}
    + \underbrace{ [\mathrm{D}y]_{0} }_{1\times 1}\,
      \underbrace{ x }_{1\times 1}
    + \frac{1}{2}
      \underbrace{ [\mathrm{D}^2y]_{0} }_{1\times 1}\,
      \underbrace{ x^{\otimes 2} }_{1\times 1}
    + \cdots\\
    &= y(0) + \frac{1}{2}\left[ \frac{\partial y}{\partial x}(0) \right]x + \frac{1}{2}\left[\frac{\partial^{2}y}{\partial x^{2}}(0)\right]x^{2} + \dots~,\notag
\end{aligned}
\end{equation}
$$

where text below the underbraces denotes the size of the corresponding tensor.

We can write out this result with the help of a Taylor expansion operator, $\mathcal{T}$, which represents its argument as a Maclaurin series:

$$
\begin{equation}
\begin{aligned}
\mathcal{T}[\mathbf{y}] = (\mathcal{T}[y_{1}]) = \left( y_{1}(0) + \frac{\partial y_{1}}{\partial x_{1}}(0) x_{1} + \frac{1}{2}\frac{\partial^{2}y_{1}}{\partial x_{1}^{2}}(0)x_{1}^{2} + \dots \right).    
\end{aligned}
\end{equation}
$$

This corresponds to a dynamical system with one physical observable described by one latent variable. Dimensionality reduction would not be applied here since we are already measuring a single quantity, the minimal number we could track.

### One Input, Multiple Outputs $(n=1, N\neq1)$

Now we consider the Taylor expansion of a univariate, vector--valued function, where $\mathbf{x}\equiv x$ and $\mathbf{0}\equiv 0$:

$$
\begin{equation}
\begin{aligned}
\underbrace{ \mathbf{y}(x) }_{ N\times 1 } &= 
\underbrace{ \mathbf{y}(0) }_{ N\times 1 } 
+ \underbrace{ [\mathrm{D}\mathbf{y}]_{0} }_{ N\times1 }\underbrace{ x }_{ 1\times 1 } 
+ \frac{1}{2}\underbrace{ [\mathrm{D}^2\mathbf{y}]_{0} }_{ N\times1 }\underbrace{ x^{\otimes 2} }_{ 1\times 1 } 
+ \ldots\\
&= 
\begin{pmatrix}
    y_1(0)\\
    y_2(0)\\
    \vdots\\
    y_N(0)
\end{pmatrix}
+
\begin{pmatrix}
\frac{\partial y_1}{\partial x}(0) \\
\frac{\partial y_2}{\partial x}(0)\\
\vdots \\
\frac{\partial y_N}{\partial x}(0)
\end{pmatrix}
x 
+ \frac{1}{2}
\begin{pmatrix}
\frac{\partial^{2} y_1}{\partial x^{2}}(0) \\
\frac{\partial^{2} y_2}{\partial x^{2}}(0)\\
\vdots \\
\frac{\partial^{2} y_N}{\partial x^{2}}(0)
\end{pmatrix}
x^{2} + \dots~.\notag
\end{aligned}
\end{equation}
$$

This shows that the Taylor expansion of a univariate, vector--valued function is a vector of Taylor expansions. We now write out the Taylor expansion of this $\mathbf{y}$ with $\mathcal{T}$:

$$
\begin{equation}
\begin{aligned}\label{eqn:taylor_vectorValuedUnivariate}
    \mathcal{T}[\mathbf{y}] = 
    \begin{pmatrix}
    \mathcal{T}[y_{1}] \\
    \mathcal{T}[y_{2}] \\
    \vdots \\
    \mathcal{T}[y_{N}]
    \end{pmatrix}
     = 
     \begin{pmatrix}
    y_{1}(0) + \frac{\partial y_{1}}{\partial x_{1}}(0)x_{1} + \frac{1}{2}\frac{\partial^{2}y_{1}}{\partial x_{1}^{2}}(0)x_{1}^{2}+\dots \\
    y_{2}(0) + \frac{\partial y_{2}}{\partial x_{1}}(0)x_{1} + \frac{1}{2}\frac{\partial^{2}y_{2}}{\partial x_{1}^{2}}(0)x_{1}^{2}+\dots  \\
    \vdots \\
    y_{N}(0) + \frac{\partial y_{N}}{\partial x_{1}}(0)x_{1} + \frac{1}{2}\frac{\partial^{2}y_{N}}{\partial x_{1}^{2}}(0)x_{1}^{2}+\dots 
    \end{pmatrix}.    
    \end{aligned}
\end{equation}
$$

Here, one could envision a dynamical system with $N$ physical observables but a singular latent variable "pulling the strings" behind the scenes. Systems well--described by a singular latent variable are rather rare but it turns out that systems described by two form an important class of dynamical systems, motivating the case of $(n\neq1,\,N\neq1)$ case we are building up to. Take note that I said referred to systems "well--described" by a singular latent variable and not systems "fully--described" by a singular latent variable. This is because a single latent variable might describe most of the dynamics---perhaps even the dynamics we actually care about---but not all of the observed data.


### Multiple Inputs, One Output $(n\neq1, N=1)$

Here we consider the Taylor expansion of a multivariate, scalar--valued function, where $\mathbf{y}\equiv y$:

$$
\begin{equation}
\begin{aligned}\label{eqn:scalar-valued,multivariate_derivation}
    \underbrace{ y(\mathbf{x}) }_{1\times 1} 
    &= \underbrace{ y(\mathbf{0}) }_{1\times 1} 
    + \underbrace{ [\mathrm{D}y]_{\mathbf{0}} }_{1\times n}\,\underbrace{ \mathbf{x} }_{n\times 1} 
    + \frac{1}{2}\underbrace{ [\mathrm{D}^2y]_{\mathbf{0}} }_{1\times n^{2}}\,\underbrace{ x^{\otimes 2} }_{n^{2}\times 1} + \ldots\\
    &= \underbrace{ y(\mathbf{0}) }_{1\times 1}
    + \underbrace{ [\mathrm{D}y]_{\mathbf{0}} }_{1\times n}\,\underbrace{ \mathbf{x} }_{n\times 1}
    + \frac{1}{2}\underbrace{ \mathbf{x}^{\top} }_{1\times n}\,\underbrace{ \left[ \mathrm{D}^{2}y \right]_{\mathbf{0}} }_{n\times n}\,\underbrace{ \mathbf{x} }_{n\times 1}
    + \ldots\notag\\
    &= y(\mathbf{0})
    + 
    \begin{pmatrix}
    \frac{\partial y}{\partial x_{1}}(\mathbf{0}) &\dots & 
    \frac{\partial y}{\partial x_{n}}(\mathbf{0})
    \end{pmatrix}
    \begin{pmatrix}
    x_{1} \\ 
    \vdots \\ 
    x_{n}
    \end{pmatrix} \\
    &\quad + \frac{1}{2}
    \begin{pmatrix}
    x_{1} & \cdots & x_{n}
    \end{pmatrix}
    \begin{pmatrix}
    \frac{\partial^{2} y}{\partial x_{1}^{2}}(\mathbf{0}) & \cdots & \frac{\partial^{2} y}{\partial x_{1}\partial x_{n}}(\mathbf{0}) \\
    \vdots & \ddots & \vdots \\
    \frac{\partial^{2} y}{\partial x_{n}\partial x_{1}}(\mathbf{0}) & \cdots & \frac{\partial^{2} y}{\partial x_{n}^{2}}(\mathbf{0})
    \end{pmatrix}
    \begin{pmatrix}
    x_{1} \\ 
    \vdots \\ 
    x_{n}
    \end{pmatrix}
    + \cdots\notag\\
    &= y(\mathbf{0})
    +
    \begin{bmatrix}
    \frac{\partial y}{\partial x_{1}}(\mathbf{0})x_{1} +\dots+
    \frac{\partial y}{\partial x_{n}}(\mathbf{0})x_{n}
    \end{bmatrix} \\
    &\quad +
    \frac{1}{2}\left[
    \frac{\partial^{2}y}{\partial x_{1}^{2}}(\mathbf{0})x_{1}^{2}
    +\dots
    +\frac{\partial^{2}y}{\partial x_{1}\partial x_{n}}(\mathbf{0})x_{1}x_{n} 
    + \frac{\partial^{2}y}{\partial x_{2}\partial x_{1}}(\mathbf{0})x_{2}x_{1} 
    + \dots \right]+\dots\notag\\
    &= y(\mathbf{0})
    + \sum_{j=1}^{m}\left[ \frac{\partial y}{\partial x_{j}}(\mathbf{0})x_{j} \right]
    + \frac{1}{2} \sum_{j=1}^{m}\sum_{k=1}^{m} \left[ \frac{\partial^{2}y}{\partial x_{j}\partial x_{k}}(\mathbf{0})x_{j}x_{k} \right]
    + \dots\notag\\
    &= y(\mathbf{0})+[\mathrm{D}y]_{\mathbf{0}}[\mathbf{x}] + \frac{1}{2}[\mathrm{D}^{2}y]_{\mathbf{0}}[\mathbf{x},\mathbf{x}]+\dots\notag,
\end{aligned}
\end{equation}
$$

where the last line leverages the notation of multilinear forms as functions of $\mathbf{x}$.

We now write out the Taylor expansion of this $\mathbf{y}$ via the Taylor expansion operator $\mathcal{T}$. Notably, we take advantage of Einstein summation notation to convey the summations involved in computing this Taylor expansion:

$$
\begin{equation}
\begin{aligned}\label{eqn:taylor_scalarValuedMultivar}
\mathcal{T}[y]=(\mathcal{T}[y_{1}]) = \left( y_{1}(\mathbf{0}) + J_{1j}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{1jk}(\mathbf{0})[x_{j},x_{k}]+\dots \right),
\end{aligned}
\end{equation}
$$

where letters $J$ and $H$ are used due to their associations with the Jacobian and Hessian, respectively.

This case is the "anti--dimensionality reduction" case, since we have multiple latent variables describing a single physical observable, so it we can only increase dimension. However, as we'll soon see, this case of $(n=1, N\neq1)$ is important for deriving the Taylor expansion of a function where $(n\neq1, N\neq1)$.

### Multiple Inputs, Multiple Outputs $(n\neq1, N\neq1)$

Finally, let's consider the Taylor expansion of a multivariate, vector--valued function. Recall that the Taylor expansion of a vector--valued function is a vector of Taylor expansions of each element. Additionally, the Taylor expansion of a multivariate, scalar--valued function can be written concisely with Einstein summation notation. Using these principles, we arrive at the following equation for the Taylor expansion of a multivariate, vector--valued function:

$$
\begin{equation}
\begin{aligned}\label{eqn:taylor_vectorValuedMultivariate}
    \mathcal{T}[\mathbf{y}] 
    = 
    \begin{pmatrix}
    \mathcal{T}[y_{1}] \\
    \mathcal{T}[y_{2}] \\
    \dots \\
    \mathcal{T}[y_{N}]
    \end{pmatrix}
    =
    \begin{pmatrix}
    y_{1}(\mathbf{0}) + G_{1j}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{1jk}(\mathbf{0})[x_{j},x_{k}] + \dots \\
    y_{2}(\mathbf{0}) + G_{2j}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{2jk}(\mathbf{0})[x_{j},x_{k}] + \dots \\
    \dots \\
    y_{N}(\mathbf{0}) + G_{Nj}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{Njk}(\mathbf{0})[x_{j},x_{k}] + \dots
    \end{pmatrix}    
\end{aligned}
\end{equation}
$$

This is the most general case and forms the mathematical foundation for inferring latent manifolds from data. Here, we have any number of latent variables describing any number of physical observables.


### Multilinear Forms

For completeness, let's also define the Maclaurin expansion of the $i$th element of a multivariate, vector--valued function with a notation more general than that previously used in this post:

$$
\begin{equation}
\begin{aligned}\label{eqn:taylor_general}
\mathcal{T}[f_{i}]&=
\frac{1}{0!}f_{i}(\mathbf{0}) + \frac{1}{1!}G_{ij}(\mathbf{0})[x_{j}] + \frac{1}{2!}H_{ijk}(\mathbf{0})[x_{j},x_{k}] + \cdots~.
\end{aligned}
\end{equation}
$$

The following table defines multilinear forms:

| Form | Mapping | Description |
|------|---------|-------------|
| $G$ | $X \to \mathbb{R}$ | Linear form |
| $H$ | $X \times X \to \mathbb{R}$ | Bilinear form |
| $F$ | $X^k \to \mathbb{R}$ | $k$–linear form |

The structure of the inferred latent manifold parameterized by a Taylor series expansion has its polynomial nonlinear structure encoded by different multilinear forms: the linear form encodes linear structure (i.e., that underpinning the subspace inferred through PCA/POD); the quadratic form encodes quadratic structure (i.e., the quadratic structure of a quadratic manifold); and so on for $k$--linear forms and higher order structure.

## Posing an Optimization Problem to Parameterize a Truncated Maclaurin Polynomial

Now we want to leverage the equation for a general Taylor series expansion when $(n\neq1, N\neq1)$ so that we can use it as a model for our underlying latent manifold. Doing so facilitates its use in an optimization problem for parameterizing the multilinear forms composing the Taylor polynomial. This optimization problem is the same one solved in PCA/POD, spectral submanifold, and polynomial manifold applications. Be aware that this subsection is rather mathy (as if this blog post wasn't already).

First, let $\mathbf{y}\in\mathbb{R}^N$ be a function of $\mathbf{x}\in\mathbb{R}^n$ and represent $\mathbf{y}$ in terms of its Maclaurin expansion. Notice that $\mathbf{y}:\mathbb{R}^N\rightarrow\mathbb{R}^n$, so $\mathbf{y}$ is a multivariate, vector--valued function, and thus permits a Maclaurin expansion: 

$$
\begin{equation}
\begin{aligned}\label{eqn:start}
    \mathbf{y}
    = 
    \begin{pmatrix}
    \mathcal{T}[y_{1}] \\
    \mathcal{T}[y_{2}] \\
    \vdots \\
    \mathcal{T}[y_{N}]
    \end{pmatrix}
    &=
    \begin{pmatrix}
    y_{1}(\mathbf{0}) + G_{1j}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{1jk}(\mathbf{0})[x_{j},x_{k}] + \dots \\
    y_{2}(\mathbf{0}) + G_{2j}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{2jk}(\mathbf{0})[x_{j},x_{k}] + \dots \\
    \vdots \\
    y_{N}(\mathbf{0}) + G_{Nj}(\mathbf{0})[x_{j}] + \frac{1}{2}H_{Njk}(\mathbf{0})[x_{j},x_{k}] + \dots
    \end{pmatrix}\\
    &=
    \begin{pmatrix}
    y_{1}(\mathbf{0}) \\
    y_{2}(\mathbf{0}) \\
    \vdots \\
    y_{N}(\mathbf{0})
    \end{pmatrix}
    +
    \begin{pmatrix}
    G_{1j}(\mathbf{0})[x_{j}] \\
    G_{2j}(\mathbf{0})[x_{j}] \\
    \vdots \\
    G_{Nj}(\mathbf{0})[x_{j}]
    \end{pmatrix}
    +\frac{1}{2}
    \begin{pmatrix}
    H_{1jk}(\mathbf{0})[x_{j},x_{k}] \\
    H_{2jk}(\mathbf{0})[x_{j},x_{k}] \\
    \vdots \\
    H_{Njk}(\mathbf{0})[x_{j},x_{k}]
    \end{pmatrix}
    +\dots~.\notag
\end{aligned}
\end{equation}
$$

We now subtract $f_{i}(\mathbf{0})$ from $\mathcal{T}[y_{i}]$, yielding $\mathcal{\tilde{T}}[y_{i}]$:

$$
\begin{equation}
\begin{aligned}
    \begin{pmatrix}
    \mathcal{T}[y_{1}]-y_{1}(\mathbf{0}) \\
    \mathcal{T}[y_{2}]-y_{2}(\mathbf{0}) \\
    \vdots \\
    \mathcal{T}[y_{N}]-y_{N}(\mathbf{0})
    \end{pmatrix}
    &=
    \begin{pmatrix}
    \mathcal{\tilde{T}}[y_{1}] \\
    \mathcal{\tilde{T}}[y_{2}] \\
    \vdots \\
    \mathcal{\tilde{T}}[y_{N}]
    \end{pmatrix} \\
    &=
    \begin{pmatrix}
    G_{1j}(\mathbf{0})[x_{j}] \\
    G_{2j}(\mathbf{0})[x_{j}] \\
    \vdots \\
    G_{Nj}(\mathbf{0})[x_{j}]
    \end{pmatrix}
    +\frac{1}{2}
    \begin{pmatrix}
    H_{1jk}(\mathbf{0})[x_{j},x_{k}] \\
    H_{2jk}(\mathbf{0})[x_{j},x_{k}] \\
    \vdots \\
    H_{Njk}(\mathbf{0})[x_{j},x_{k}]
    \end{pmatrix}
    +\dots~.
\end{aligned}
\end{equation}
$$

To improve legibility, we define $\tilde{\mathbf{y}}$ as the corresponding vector of these $\mathcal{\tilde{T}}[y_{i}]$:

$$
\begin{equation}
\begin{aligned}
    \tilde{\mathbf{y}}
    =
    \begin{pmatrix}
    G_{1j}(\mathbf{0})[x_{j}] \\
    G_{2j}(\mathbf{0})[x_{j}] \\
    \vdots \\
    G_{Nj}(\mathbf{0})[x_{j}]
    \end{pmatrix}
    +\frac{1}{2}
    \begin{pmatrix}
    H_{1jk}(\mathbf{0})[x_{j},x_{k}] \\
    H_{2jk}(\mathbf{0})[x_{j},x_{k}] \\
    \vdots \\
    H_{Njk}(\mathbf{0})[x_{j},x_{k}]
    \end{pmatrix}
    +\dots~.
\end{aligned}
\end{equation}
$$


Next, we "pull out" the arguments of the multilinear forms into their corresponding vectors of monomials. Here, $G_{i}(\mathbf{0};j)$ is a function parameterized by index $j$ corresponding to the $j$ in $x_{j}$
Likewise for $H_{i}(\mathbf{0};j,k)$ with indices $j$ and $k$ from product $x_{j}x_{k}$ and higher--order terms.

$$
\begin{equation}
\begin{aligned}\label{eqn:pull-out}
    \tilde{\mathbf{y}}
    =
    \begin{pmatrix}
    G_{1}(\mathbf{0};j) \\
    G_{2}(\mathbf{0};j) \\
    \vdots \\
    G_{N}(\mathbf{0};j)
    \end{pmatrix}
    \begin{pmatrix}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{n}
    \end{pmatrix}
    +
    \begin{pmatrix}
    H_{1}(\mathbf{0};j,k) \\
    H_{2}(\mathbf{0};j,k) \\
    \vdots \\
    H_{N}(\mathbf{0};j,k)
    \end{pmatrix}
    \begin{pmatrix}
    x_{1}^{2} \\
    x_{1}x_{2} \\
    x_{2}x_{1} \\
    x_{1}x_{3} \\
    \vdots \\
    x_{n}x_{n-1} \\
    x_{n}^{2}
    \end{pmatrix}
    +\dots~,
\end{aligned}
\end{equation}
$$

where the factor of $1/2$ was absorbed into each $H_i(\mathbf{0};j,k)$ to simplify notation.

To better understand this equation, we now write out the elements of its matrix operators explicitly:

$$
\begin{equation}
\begin{aligned}\label{eqn:ytilde_matrices}
    \tilde{\mathbf{y}}
    &=
    \begin{pmatrix}
    G_{1}(\mathbf{0};1) & G_{1}(\mathbf{0};2) & \dots & G_{1}(\mathbf{0};n) \\
    G_{2}(\mathbf{0};1) & G_{2}(\mathbf{0};2) & \dots & G_{2}(\mathbf{0};n) \\
    \vdots \\
    G_{N}(\mathbf{0};1) & G_{N}(\mathbf{0};2) & \dots & G_{N}(\mathbf{0};n) \\
    \end{pmatrix}
    \begin{pmatrix}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{n}
    \end{pmatrix}+
    \\&+
    \begin{pmatrix}
    H_{1}(\mathbf{0};1,1)  & H_{1}(\mathbf{0};1,2)  & H_{1}(\mathbf{0};2,1)  & H_{1}(\mathbf{0};2,2)  & \dots  & H_{1}(\mathbf{0};n,n-1)  & H_{1}(\mathbf{0};n,n)\\
    H_{2}(\mathbf{0};1,1)  & H_{2}(\mathbf{0};1,2)  & H_{2}(\mathbf{0};2,1)  & H_{2}(\mathbf{0};2,2)  & \dots  & H_{2}(\mathbf{0};n,n-1)  & H_{2}(\mathbf{0};n,n) \\
    \vdots \\
    H_{N}(\mathbf{0};1,1)  & H_{N}(\mathbf{0};1,2)  & H_{N}(\mathbf{0};2,1)  & H_{1}(\mathbf{0};2,2)  & \dots  & H_{N}(\mathbf{0};n,n-1)  & H_{N}(\mathbf{0};n,n)
    \end{pmatrix}
    \begin{pmatrix}
    x_{1}^{2} \\
    x_{1}x_{2} \\
    x_{2}x_{1} \\
    x_{1}x_{3} \\
    \vdots \\
    x_{n}x_{n-1} \\
    x_{n}^{2}
    \end{pmatrix}
    +\dots~.\notag
\end{aligned}
\end{equation}
$$

We can thus deduce the sizes of the respective matrix operators, where Kronecker product notation and matrices $\mathbf{T}_1$ and $\mathbf{T}_2$ have been leveraged for conciseness:

$$
\begin{equation}
\begin{aligned}\label{eqn:ytilde_sizes}
\underbrace{ \tilde{\mathbf{y}} }_{ \mathbb{R}^{N\times 1} }
&=
\underbrace{ \begin{pmatrix}
G_{1}(\mathbf{0};j) \\
G_{2}(\mathbf{0};j) \\
\vdots \\
G_{N}(\mathbf{0};j)
\end{pmatrix} }_{ \mathbb{R}^{N\times n} }
\underbrace{ \mathbf{x} }_{ \mathbb{R}^{n\times 1} }
+
\underbrace{ \begin{pmatrix}
H_{1}(\mathbf{0};j,k) \\
H_{2}(\mathbf{0};j,k) \\
\vdots \\
H_{N}(\mathbf{0};j,k)
\end{pmatrix} }_{ \mathbb{R}^{N\times n^{2}} }
\underbrace{ \mathbf{x}\otimes\mathbf{x} }_{ \mathbb{R}^{n^{2}\times 1} }
+\dots\\
&=\mathbf{T}_1\mathbf{x}
+
\mathbf{T}_2\mathbf{x}\otimes\mathbf{x}+\dots\notag~.
\end{aligned}
\end{equation}
$$


So far we have considered an infinite series Maclaurin expansion. Let us now consider the following truncated expansion, yielding a quadratic approximation of $\tilde{\mathbf{y}}$:

$$
\begin{equation}
\begin{aligned}\label{eqn:truncated}
\tilde{\mathbf{y}}
\approx
\mathbf{T}_1\mathbf{x}
+
\mathbf{T}_2\mathbf{x}\otimes\mathbf{x}.
\end{aligned}
\end{equation}
$$

This truncation is necessary since our end goal is to solve a problem on a computer, so we cannot infer an infinite amount of terms. However, it does not mean that our results won't be accurate, as truncations such that the largest $d$ is $1$ are common---they, in fact, formulate linear PCA/POD!

Assuming that we have access to numerical data sufficiently describing $\tilde{\mathbf{y}}$ and $\mathbf{x}$, we can thus pose the following least--squares optimization problem to infer the values of $G_{i}(\mathbf{0};j)$ and $H_{i}(\mathbf{0};j,k)$ where $i\in \{ 1,2,\dots,N \}$ and $j,k\in\{1,2,\dots,n\}$:

$$
\begin{equation}
\begin{aligned}
\text{argmin}_{\mathbf{T}_1,\mathbf{T}_2}
\left\lVert  
\tilde{\mathbf{y}}
-
\mathbf{T}_1\mathbf{x}
-
\mathbf{T}_2\mathbf{x}\otimes\mathbf{x}
\right\rVert_2^2.
\end{aligned}
\end{equation}
$$


Of course, we could generalize this optimization problem to include monomial terms in our Kronecker product vectors up to degree $d$, yielding the following problem:

$$
\begin{equation}
\begin{aligned}\label{eqn:result}
    \text{argmin}_{\{ \mathbf{T}_{p} \}}
    \left\lVert  
    \tilde{\mathbf{y}}
    - \sum_{p=1}^d\mathbf{T}_p\mathbf{x}^{\otimes p}
    \right\rVert_2^2.
\end{aligned}
\end{equation}
$$

Next, we can assume that $\mathbf{T}_p=\mathbf{V}_p\mathbf{R}_p$, where $\mathbf{Q}_p\in\mathbb{R}^{N\times n^p}$ is orthogonal and $\mathbf{R}_p\in\mathbb{R}^{n^p\times n^p}$ is upper--triangular. Using this QR decomposition, we can  rewrite our truncated expansion using orthogonal matrices, where $\mathbf{s}^{\otimes p}\equiv\mathbf{R}_p\mathbf{x}^{\otimes p}$:

$$
\begin{equation}
\begin{aligned}\label{eqn:result_with_orthogonal_matrices}
    \text{argmin}_{\{ \mathbf{V}_{p} \}}
    \left\lVert  
    \tilde{\mathbf{y}}
    - \sum_{p=1}^d\mathbf{V}_p\mathbf{s}^{\otimes p}
    \right\rVert_2^2.
\end{aligned}
\end{equation}
$$

Finally, given that we use multiple samples of the dynamical system's physical observables, we need to perform this optimization with respect to all samples simultaneously. This is done storing all samples as columns in matrices:

$$
\begin{equation}
\begin{aligned}
    \left( \mathbf{V}_1, \dots, \mathbf{V}_d \right) = \text{argmin}_{\{ \mathbf{V}_{p} \}}
    \left\lVert  
    \tilde{\mathbf{Y}}
    - \sum_{p=1}^d\mathbf{V}_p\mathbf{S}_K^{\odot p}
    \right\rVert_F^2,
\end{aligned}
\end{equation}
$$

where $\tilde{\mathbf{Y}} = \begin{bmatrix} \tilde{\mathbf{y}}_1 & \cdots & \tilde{\mathbf{y}}_K \end{bmatrix}$ and $\mathbf{S}^{\odot p} = \begin{bmatrix} \mathbf{s}_1^{\otimes p} & \cdots & \mathbf{s}_K^{\otimes p} \end{bmatrix}$, and $K$ is the number of samples (aka, snapshots).


## Sequential Closure Modeling

One way to solve for decoders $\{\mathbf{V}\_1, \mathbf{V}\_2, \dots, \mathbf{V}\_d\}$ is to do so sequentially: using reconstruction error with respect to decoders $\{\mathbf{V}\_1, \dots, \mathbf{V}\_{p-1}\}$ to formulate a linear least–squares optimization problem for inferring decoder $\mathbf{V}\_p$.

### Inferring $\mathbf{V}_1$

To begin our sequential optimization we must infer our linear decoder $\mathbf{V}_1$. The optimization for $\mathbf{V}_1$ is as follows

$$
\begin{equation}
\begin{aligned}\label{eqn:V1_min_problem}
    \mathbf{V}_1 = \text{argmin}\frac{1}{2}\lVert \mathbf{V}_1^\top \mathbf{S} - \hat{\mathbf{\mathbf{S}}} \rVert_\text{F}^2.
\end{aligned}
\end{equation}
$$

The solution to this least--squares problem is given by the matrix of POD basis vectors obtained through the SVD.

With linear decoder $\mathbf{V}_1$ now in hand we can compute the reconstruction error associated with it, $\mathcal{E}_1$:

$$
\begin{equation}
\begin{aligned}\label{eqn:err_after_linear}
    \mathcal{E}_1&=(\mathbf{I}-\mathbf{V}_1\mathbf{V}_1^\top)\mathbf{S}\nonumber\\
    &= [\tilde{\mathbf{V}}_1][\tilde{\mathbf{V}}_1]^\top \mathbf{S}.
\end{aligned}
\end{equation}
$$

This reconstruction error with respect to $\mathbf{V}_1$ can then be used as the basis to infer $\mathbf{V}_2$.

### Inferring $\mathbf{V}_2$

We can leverage $\mathcal{E}_1$ to formulate an optimization problem for $\mathbf{V}_2$ such that it best fixes the error left over from $\mathbf{V}_1$. This is written out in the following equation, where $\mathbf{W}_2=\mathbf{S}^{\odot 2} = \mathbf{S}\odot\mathbf{S}=\begin{pmatrix}
    \mathbf{s}_1\otimes\mathbf{s}_1 \dots \mathbf{s}_K\otimes \mathbf{s}_K
\end{pmatrix}$ is the second Khatri--Rao power of $\mathbf{S}$, storing all quadratic snapshot monomials of all snapshots:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_min_problem}
    \mathbf{V}_2 = \text{argmin}\frac{1}{2}\lVert W_2^\top \mathbf{V}_2^\top - \mathcal{E}_1^\top \rVert_\text{F}^2
\end{aligned}
\end{equation}
$$

Computing $\mathbf{V}_2$ as the quadratic decoder that best--closes the linear model parameterized by $\mathbf{V}_1$ establishes that $\mathbf{V}_2~\perp~\mathbf{V}_1$. We will now derive this.

First we define the loss function, $J$, that we intend to optimize. For optimizing $\mathbf{V}_2$ through our sequential optimization framework, $J=J(\mathbf{V}_2)$. We are seeking the least--squares solution and thus obtain the following loss function:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_cost_function}
    J(\mathbf{V}_{2}) &= \frac{1}{2}\lVert \mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top}-\mathcal{E}_{1}^{\top} \rVert_\text{F}^2,
\end{aligned}
\end{equation}
$$

where the factor of $1/2$ in front of the Frobenius norm exists to simplify the arithmetic encountered in matrix calculus--based optimization.

With an equation for $J(\mathbf{V}_{2})$ in hand we can put it into an equivalent but more suitable form for optimization by taking advantage of the trace operator, $\text{tr}$, as done as follows:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_cost_function_trace}
    \frac{1}{2}\lVert \mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top}-\mathcal{E}_{1}^{\top} \rVert_\text{F}^2 &= \frac{1}{2}\text{tr}\left( \left( \mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top}-\mathcal{E}_{1}^{\top} \right)^{\top} \left( \mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top}-\mathcal{E}_{1}^{\top} \right)  \right)  \nonumber\\
    &=\frac{1}{2}\text{tr} \left( \left( \mathbf{V}_{2}\mathbf{W}_{2}-\mathcal{E}_{1} \right) \left( \mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top}-\mathcal{E}_{1}^{\top} \right)  \right) \nonumber \\
    &= \frac{1}{2}\text{tr}\left( \mathbf{V}_{2}\mathbf{W}_{2}\mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top} - \mathbf{V}_{2}\mathbf{W}_{2}\mathcal{E}_{1}^{\top} - \mathcal{E}_{1}\mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top} + \mathcal{E}_{1}\mathcal{E}_{1}^{\top} \right) \nonumber \\
    &= \frac{1}{2}\text{tr}\left( \mathbf{V}_{2}\mathbf{W}_{2}\mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top} \right) - \text{tr}\left( \mathbf{V}_{2}\mathbf{W}_{2}\mathcal{E}_{1}^{\top} \right) + \frac{1}{2}\text{tr}\left( \mathcal{E}_{1}\mathcal{E}_{1}^{\top} \right) 
\end{aligned}
\end{equation}
$$

Our goal is to minimize the loss function $J(\mathbf{V}_2)$ with respect to the variable $\mathbf{V}_2$, so we will now compute the gradient of $J(\mathbf{V}_2)$ with respect to $\mathbf{V}_2$:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_gradient}
    \nabla_{\mathbf{V}_{2}}J(\mathbf{V}_{2}) &= \frac{1}{2}\nabla_{\mathbf{V}_{2}}\left( \mathbf{V}_{2}\mathbf{W}_{2}\mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top} \right) - \nabla_{\mathbf{V}_{2}}\left( \mathbf{V}_{2}\mathbf{W}_{2}\mathcal{E}_{1}^{\top} \right) \\
    &\quad + \cancel{ \frac{1}{2}\nabla_{\mathbf{V}_{2}}\text{tr}\left( \left( \mathbf{I}-\mathbf{V}_{1}\mathbf{V}_{1}^{\top} \right) S_{1}S_{1}^{\top}\left( \mathbf{I}-\mathbf{V}_{1}\mathbf{V}_{1}^{\top} \right)  \right) }  \nonumber \\
    &= \mathbf{W}_{2}\mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top} - \mathbf{W}_{2}\mathcal{E}_{1}^{\top},
\end{aligned}
\end{equation}
$$

where we can see that including the factor of $1/2$ in $J(\mathbf{V}\_2)$ allows us to have no non--unity scalars in our expression of the gradient in the equation for $\nabla_{\mathbf{V}\_{2}}J(\mathbf{V}\_{2})$.

Now that we have $\nabla_{\mathbf{V}\_{2}}J(\mathbf{V}\_{2})$ we can force it to be $0$ so that we identify a root of $\nabla_{\mathbf{V}\_2}J(\mathbf{V}\_2)$ and thus an extremum of $J(\mathbf{V}\_2)$:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_optimization_via_gradient}
    \nabla_{\mathbf{V}_{2}}J(\mathbf{V}_{2}) \overset{!}{=} 0 \implies \mathbf{W}_{2}\mathbf{W}_{2}^{\top}\mathbf{V}_{2}^{\top} - \mathbf{W}_{2}\mathcal{E}_{1}^{\top} = 0,
\end{aligned}
\end{equation}
$$

where $\overset{!}{=}$ notation is used since we are forcing $\nabla_{\mathbf{V}_2}J(\mathbf{V}_2)$ to equal $0$ for optimization purposes.

Finally, we can solve for $\mathbf{V}_2$ to show that $\mathbf{V}_2~\perp~\mathbf{V}_1$:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_orthogonal_to_V1}
    \mathbf{V}_{2}^{\top} &= \left(\mathbf{W}_{2}\mathbf{W}_{2}^{\top}\right)^{-1}\mathbf{W}_{2}\mathcal{E}_{1}^{\top} \nonumber \\
    \mathbf{V}_{2} &= \mathcal{E}_{1}\mathbf{W}_{2}^{\top}\left( \mathbf{W}_{2}\mathbf{W}_{2}^{\top} \right)^{-1} \nonumber  \\
    &= (\mathbf{I}-\mathbf{V}_{1}\mathbf{V}_{1}^{\top}) \mathbf{S}\mathbf{W}_{2}^{\top}\left( \mathbf{W}_{2}\mathbf{W}_{2}^{\top} \right)^{-1}\nonumber \\
    &= \tilde{\mathbf{V}}_{1}\tilde{\mathbf{V}}_{1}^{\top}\left( \mathbf{S} \mathbf{W}_{2}^{\top}\left( \mathbf{W}_{2}\mathbf{W}_{2}^{\top} \right)^{-1}  \right),
\end{aligned}
\end{equation}
$$

where $\tilde{\mathbf{V}}_1\tilde{\mathbf{V}}_1^\top$ is an orthogonal projector onto the orthogonal complement of $\mathbf{V}_1$. Therefore, $\mathbf{V}_2~\perp~ \mathbf{V}_1$.

Additionally, we can compute the reconstruction error after leveraging both the linear and quadratic decoder, $\mathcal{E}_2$:

$$
\begin{equation}
\begin{aligned}\label{eqn:err_after_quadratic}
    \mathcal{E}_{2} &= \left( \mathbf{I}-\mathbf{V}_{1}\mathbf{V}_{1}^{\top}-\mathbf{V}_{2}\mathbf{V}_{2}^{\top} \right) \mathbf{S} \nonumber\\
    &= \left[ \tilde{\mathbf{V}}_{1} \tilde{\mathbf{V}}_{2} \right] \left[ \tilde{\mathbf{V}}_{1} \tilde{\mathbf{V}}_{2} \right]^{\top} \mathbf{S}.
\end{aligned}
\end{equation}
$$

Our goal so far, however, is to infer the $\mathbf{V}_2$ that minimizes its given loss function. Since the optimization problem is a linear least--squares problem, we can solve it with the QR decomposition as follows:

$$
\begin{equation}
\begin{aligned}\label{eqn:V2_least_squares}
    \mathbf{V}_{2}\mathbf{W}_{2}\mathbf{W}_{2}^{\top} &= \mathcal{E}_{1}\mathbf{W}_{2}^{\top}\nonumber \\
    \mathbf{V}_{2}\left( \mathbf{Q}_{2}\mathbf{R}_{2} \right)^{\top}\left( \mathbf{Q}_{2}\mathbf{R}_{2} \right) &= \mathcal{E}_{1}(\mathbf{Q}_{2}\mathbf{R}_{2}) \nonumber\\
    \mathbf{V}_{2}\mathbf{R}_{2}^{\top}\mathbf{R}_{2} &= \mathcal{E}_{1}\mathbf{Q}_{2}\mathbf{R}_{2} \nonumber\\
    \mathbf{V}_{2}\mathbf{R}_{2}^{\top} &= \underbrace{ \mathcal{E}_{1}\mathbf{Q}_{2} }_{ \mathbf{Y}_{2} } \nonumber\\
    \mathbf{R}_{2}\mathbf{V}_{2}^{\top} &= \mathbf{Y}_{2}^{\top}.
\end{aligned}
\end{equation}
$$

### Inferring $\mathbf{V}_3$

The cubic decoder $\mathbf{V_3}$ can be inferred as the matrix that best--fixes the quadratic reconstruction error by solving the following optimization problem:

$$
\begin{equation}
\begin{aligned}
    \mathbf{V}_{3}=\text{argmin}\frac{1}{2}\lVert \mathbf{W}_{3}^{\top}\mathbf{V}_{3}^{\top}-\mathcal{E}_{2}^{\top} \rVert_\text{F}^2,
\end{aligned}
\end{equation}
$$

where $\mathbf{W}_3 = \mathbf{S}^{\diamond3} =  \mathbf{S}\odot \mathbf{S}\odot \mathbf{S} = (\mathbf{s}_1\otimes\mathbf{s}_1\otimes\mathbf{s}_1\dots\mathbf{s}_K\otimes\mathbf{s}_K\otimes\mathbf{s}_K)$ is the third Khatri--Rao power of $\mathbf{S}$ that stores all cubic monomial terms of all snapshots.

We can prove that $\mathbf{V}_3$ is orthogonal to both $\mathbf{V_1}$ and $\mathbf{V_2}$:

$$
\begin{equation}
\begin{aligned}
    \mathbf{V}_{3} &= \mathcal{E}_{2}\mathbf{W}_{3}^{\top}\left( \mathbf{W}_{3}\mathbf{W}_{3}^{\top} \right)^{-1} \nonumber\\
    &= \left( \mathbf{I}- \mathbf{V}_{1}\mathbf{V}_{1}^{\top} - \mathbf{V}_{2}\mathbf{V}_{2}^{\top} \right)\mathbf{S}\mathbf{W}_{3}^{\top}\left( \mathbf{W}_{3}\mathbf{W}_{3}^{\top} \right)^{-1} \nonumber\\
    &= \left[ \tilde{\mathbf{V}}_{1} \tilde{\mathbf{V}}_{2} \right] \left[ \tilde{\mathbf{V}}_{1} \tilde{\mathbf{V}}_{2} \right]^{\top} \left( \mathbf{S}\mathbf{W}_{3}^{\top}\left( \mathbf{W}_{3}\mathbf{W}_{3}^{\top} \right)^{-1} \right) 
\end{aligned}
\end{equation}
$$

where $\left[ \tilde{\mathbf{V}}\_{1} \tilde{\mathbf{V}}\_{2} \right] \left[ \tilde{\mathbf{V}}\_{1} \tilde{\mathbf{V}}\_{2} \right]^{\top}$ is a projector onto the orthogonal complement of $\text{span}\{ \mathbf{v}^{(1)}\_1, \dots, \mathbf{v}^{(n)}\_1, \mathbf{v}^{(1)}\_2, \dots, \mathbf{v}^{(n^2)}\_1 \}$. Therefore, $\mathbf{V}\_3 ~\perp~ \{\mathbf{V}\_1, \mathbf{V}\_2\}$.

Additionally, we can identify a closed--form solution to the least--squares problem for $\mathbf{V}_3$ using the QR decomposition:

$$
\begin{equation}
\begin{aligned}\label{eqn:V3_least_squares}
    \mathbf{V}_{3}\mathbf{W}_{3}\mathbf{W}_{3}^{\top} &= \mathcal{E}_{2}\mathbf{W}_{3}^{\top} \nonumber\\
    \mathbf{V}_{3}\left( \mathbf{Q}_{3}\mathbf{R}_{3} \right)^{\top}\left( \mathbf{Q}_{3}\mathbf{R}_{3} \right) &= \mathcal{E}_{2}(\mathbf{Q}_{3}\mathbf{R}_{3}) \nonumber\\
    \mathbf{V}_{3}\mathbf{R}_{3}^{\top}\mathbf{R}_{3} &= \mathcal{E}_{2}\mathbf{Q}_{3}\mathbf{R}_{3}\nonumber \\
    \mathbf{V}_{3}\mathbf{R}_{3}^{\top} &= \underbrace{ \mathcal{E}_{2}\mathbf{Q}_{3} }_{ \mathbf{Y}_{3} } \nonumber\\
    \mathbf{R}_{3}\mathbf{V}_{3}^{\top} &= \mathbf{Y}_{3}^{\top}.
\end{aligned}
\end{equation}
$$


### Inferring $\mathbf{V}_{k+1}$

Finally, we may apply logic analogous to that used to derive $\mathbf{V}\_2$ from $\mathbf{V}\_1$ and $\mathbf{V}\_3$ from $\{\mathbf{V}\_1,\mathbf{V}\_2\}$ in order to derive $\mathbf{V}\_{k+1}$ from $\{\mathbf{V}\_k\}\_{k=1}^d$.

We begin by posing the following least--squares problem for $\mathbf{V}_{k+1}$:

$$
\begin{equation}
\begin{aligned}
    \mathbf{V}_{k+1} = \text{argmin}\frac{1}{2}\lVert \mathbf{W}_{k+1}^{\top}\mathbf{V}_{k+1}^{\top}-\mathcal{E}_{k}^{\top} \rVert_{\text{F}}^2,
\end{aligned}
\end{equation}
$$

where $\mathbf{W}\_{k+1}=\mathbf{S}^{\odot(k+1)}$ is the $(k+1)$th Khatri--Rao power of $\mathbf{S}$ and $\mathcal{E}\_k$ is the reconstruction error with respect to decoders $\{ \mathbf{V}\_p \}\_{p=1}^{k}$.

Next, we can prove that decoder $\mathbf{V}\_{k+1}$ is orthogonal to $\{ \mathbf{V}\_1, \dots, \mathbf{V}\_{k}\}$:

$$
\begin{equation}
\begin{aligned}
    \mathbf{V}_{k+1} &= \mathcal{E}_{k}\mathbf{W}_{k+1}^{\top}\left( \mathbf{W}_{k+1}\mathbf{W}_{k+1}^{\top} \right)^{-1}\nonumber \\
    &= \left( \mathbf{I}- \sum_{i=1}^{k}\mathbf{V}_{i}\mathbf{V}_{i}^{\top} \right) \mathbf{S} \mathbf{W}_{k+1}^{\top}\left( \mathbf{W}_{k+1}\mathbf{W}_{k+1}^{\top} \right)^{-1} \nonumber \\
    &= \left[ \tilde{\mathbf{V}}_{1} \tilde{\mathbf{V}}_{2} \dots \tilde{\mathbf{V}}_{k} \right] \left[ \tilde{\mathbf{V}}_{1} \tilde{\mathbf{V}}_{2} \dots \tilde{\mathbf{V}}_{k} \right]^{\top} \left(\mathbf{S} \mathbf{W}_{k+1}^{\top}\left( \mathbf{W}_{k+1}\mathbf{W}_{k+1}^{\top} \right)^{-1} \right),
\end{aligned}
\end{equation}
$$

such that $\mathbf{V}_{k+1}~\perp~ \{ \mathbf{V}_1,\mathbf{V}_2,\dots,\mathbf{V}_k \}$.

Lastly, we can solve a system of linear equations to infer $\mathbf{V}_{k+1}$, where the derivation is analogous to those used to derive expressions for $\mathbf{V}_1$ and $\mathbf{V}_2$:

$$
\begin{equation}
\begin{aligned}
    \mathbf{R}_{k+1}\mathbf{V}_{k+1}^{\top} = \mathbf{Y}_{k+1}^\top.
\end{aligned}
\end{equation}
$$


## Conclusion

In this post we have shown how to solve an optimization problem from data to infer latent manifolds modeled by truncated Taylor series expansions. Such manifolds form the basis for both polynomial manifolds and spectral submanifolds in model order reduction. In fact, Justin suspects that they are different names of the same optimization result.