---
layout: page
title: 2D Turbulence with Neural Operators
description: Chaining together numerical and data-driven methods 
img: assets/img/turb_streaks.png
importance: 1
category: 
related_publications: false
---

Viscosity of liquids varies significantly with temperature, altering key features in turbulent flows. Direct Numerical Simulations (DNS) of such flows, however,
can be prohibitively expensive, spanning weeks on supercomputing clusters. We are using reduced order models and analytical methods to efficiently compute statistics in 
wall-bounded viscosity-stratified turbulence. 

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/vsf_micro-2.png" title="Governing equations" class="img-fluid rounded z-depth-1 w-100" %}
    </div>
</div>
<div class="caption text-center">
    System of equations for viscosity stratified flows. The plot on the right shows that viscosity of water can vary by a factor of two or more under commonly occuring temperature differences.
</div>

The effects of viscosity stratification can be peculiar. The image below shows that there is an excess of turbulent kinetic energy near the more viscous side of the channel relative to the less viscous side. 

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/vsf_micro.png" title="Turbulent Kinetic Energy" class="img-fluid rounded z-depth-1 w-100" %}
    </div>
</div>
<div class="caption text-center">
    Turbulent Kinetic Energy is subdued near the top (less viscous) wall and increased near the bottom (more viscous) wall.
</div>

Viscosity stratification is common in several engineering applications such as cooling systems in electronic devices. We are studying its consequences towards heat transfer in such systems.





