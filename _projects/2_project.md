---
layout: page
title: 2D Turbulence with Neural Operators
description: Chaining together numerical and data-driven methods 
img: assets/img/FNO-1.png
importance: 1
category: 
related_publications: false
---

Direct numerical simulations (DNS) are an accurate but expensive method for solving systems of Partial Differential Equations (PDEs). Data-driven technologies, on the other hand, can be much faster but struggle to maintain their accuracy while simulating chaotic systems. In this work we first train Fourier Neural Operators (FNOs) using DNS data and then use FNO in tandem with DNS to achieve accurate simulations at nearly half the computational cost. 

We demostrate a machine learning workflow that alternates between FNO and the PDE solver by using the output of FNO as input to the PDE solver and vice versa. Trade-off between error accumulation over Lyapunov time scales and computational efficiency favored hybrid approaches over purely traditional or data-driven approaches. The image shown below emphasizes the accuracy of hybrid approaches over purely data-driven methods.


<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/FNO-1.png" class="img-fluid rounded z-depth-1 w-100" %}
    </div>
</div>
<div class="caption text-center">
    Time evolution of vorticity fields from DNS , FNO and hybrid approaches. 
</div>


<p>
    This work was presented at <a href="https://sc24.supercomputing.org/program/proceedings-archives/" target="_blank">SC24</a>.  
    You can <a href="https://arxiv.org/pdf/2409.14660" target="_blank">view the paper on ArXiV</a>.
</p>

Particle-resolved numerical simulations were performed in C++ and PyTorch’s neural operator library was used for FNO.

