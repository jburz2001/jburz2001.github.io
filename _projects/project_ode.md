---
layout: page
title: ODE Simulation
description: MATLAB-based simulation of ordinary differential equations
img: assets/img/ode.png
importance: 1
category: Kramer Research Group
---

I worked with Dr. Boris Kramer and Dr. Harsh Sharma in using machine learning to derive reduced Lagrangians from data. To do so, we needed data, so I used MATLAB to model the dynamics of a mass-spring-damper and linear beam.

## Mass-Spring-Damper (MSD)

### Background

<div class="col-sm mt-3 mt-md-0">
    {% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

The MSD is a cannonical mechanical system composed of masses that are connected via restorative springs and dissipative dampers. This system can model [actual masses connected to springs and dampers](https://www.motioncontroltips.com/what-is-a-tuned-mass-damper-and-how-is-it-used-in-motion-control/), but it also serves as a useful toy model for systems such as [buildings](https://link.springer.com/article/10.1007/s10518-020-00973-2), [turbulent fluids](https://link.springer.com/article/10.1007/s11071-018-04749-x), and even [biological tissue](https://www.semanticscholar.org/paper/Simulation-of-Biological-Tissue-using-Models-Eriksson/dc9f974969e8a0e062b7575bfe852b0e591697b9).

### Simulation

...

## Linear 1-Dimensional Beam

...

### Background

...

### Simulation

...

To give your project a background in the portfolio page, just add the img tag to the front matter like so:

    ---
    layout: page
    title: project
    description: a project with a background image
    img: /assets/img/12.jpg
    ---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, *bled* for your project, and then... you reveal it's glory in the next row of images.


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```

{% endraw %}
