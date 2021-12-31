---
layout: page
title: Genetic Programming
description: GPTIPS2-based symbolic regression
img: assets/img/gp.png
importance: 2
category: Kramer Research Group
---

I worked with Dr. Boris Kramer and Dr. Harsh Sharma in using machine learning to derive reduced Lagrangians from data. The seminal machine learning technique was genetic programming, so I used MATLAB's [GPTIPS2](https://sites.google.com/site/gptips4matlab/) with the goal of emulating the original work on machine learning-based physics discovery.

## What Is Genetic Programming (GP) ?

GP is a machine learning technique used to construct computer programs through natural selection. This is done through mutation, inheritance, and other genetic operations that iteratively act upon a population of computer programs until a sufficiently fit individual is produced. 

## What Is Symbolic Regression ?

Mathematical equations can be written in computer code. Therefore, we can use GP to evolve mathematical equations to a desired result, such as a regression model for a dataset. Constructing this regression equation through GP invovles combining elementary functions (e.g. $x^2$, $sin(x)$) until the model has a sufficient $R^2$ value.

## GPTIPS2

I used GPTIPS2 software (which is based on MATLAB) to perform symbolic regression on custom datasets. 



{% endraw %}
