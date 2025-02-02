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

![FNO_comparisons](/assets/img/FNO-1.png)

This work was presented at ![SC24](https://sc24.supercomputing.org/program/proceedings-archives/). You can ![view the paper on ArXiV](https://arxiv.org/pdf/2409.14660)

Particle-resolved numerical simulations were performed in C++ and PyTorch’s neural operator library was used for FNO.

