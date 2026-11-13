---
layout: post
title: Good and Bad Gradients for Climate Control
date: 2026-11-12 12:00:00
description: When does backpropagation through a climate model give a usable control signal?
tags: climate geoengineering
categories: research
related_posts: false
toc:
  sidebar: left
---

> **In short.** Gradient-based climate control is only as good as its gradients. We brighten ocean clouds in a differentiable climate model and ask whether the backpropagated gradient of a target quantity is usable. For **global-mean SST** it matches finite differences out to 32 days. For **land temperature error** it is noisy even at a few days, and for **land precipitation error** it carries almost no usable signal. A simple signal-to-noise ratio, the predicted change over one step against the weather noise of the target, sorts usable actuator-target pairs from unusable ones.
{: .block-tip }

## Why gradients matter

Our [receding horizon control](/blog/2026/receding-horizon-climate-control/) optimizes forcing patterns by backpropagating through a few days of a differentiable climate model. That only works if the gradient of the objective points somewhere useful. Two things can go wrong:

- **Chaos.** Over long horizons, a single-trajectory gradient grows with the sensitivity of the weather, not of the climate, and stops matching the response of the time-mean objective.
- **Noise.** Even when the gradient is correct, the change it predicts can be smaller than the weather noise of the objective. A step along it then does not reliably improve the objective.

Here we probe both, for one actuator and three targets.

## The experiment

- **Model.** JAX-GCM (jcm 3) coupled to a 50 m slab ocean and a slab land surface, so SST and land temperature respond to forcing.
- **Actuator.** Stratiform cloud albedo over the ocean, `albcls` $$= 0.5 + s$$, a crude stand-in for marine cloud brightening. We sweep $$s$$ from 0 to 0.5 on 41 points.
- **Targets** $$J(s)$$, each averaged over a window of $$H$$ = 4, 8, 16, 32 or 64 days:
  - global-mean SST, in the 1×CO₂ control climate;
  - mean-square error of the land T2m map against the 1×CO₂ control run on the same dates, in a 4×CO₂ climate;
  - the same for land precipitation.
- **Gradients.** Backpropagated $$dJ/ds$$ at every 5th point, compared with finite-difference (FD) chords of width $$D = 0.025$$ and $$D = 0.25$$ taken from the sweep.
- **Ensemble.** 5 members, started on 1 July of 5 different years. Curves show the 5-member mean.

How to read each figure: one row per horizon. The left panel is the sweep $$J(s)$$ with the backpropagated tangents drawn on it; the middle and right panels compare the backpropagated slope (line) with FD chords (dots). A usable gradient has tangents that follow $$J(s)$$ and a slope that agrees with the wide chords.

## Good gradients: global-mean SST

{% include theme_figure.liquid dir="grad_web" stem="fig1_sst_global_slice" title="Global-mean SST vs ocean cloud albedo" %}
<div class="caption">
    Global-mean SST, 1×CO₂ control climate, 5-member mean. Rows: horizon H. Left: J(s) − J(0) with backpropagated tangents. Middle, right: backpropagated dJ/ds vs FD chords of width 0.025 and 0.25.
</div>

- $$J(s)$$ is a clean straight line, and the tangents lie on it through 32 days.
- The wide chords match backprop to within **0.4%** at every horizon up to 32 days (chord / backprop 0.996–1.002).
- Narrow chords scatter: over a step of 0.025 the weather noise is comparable to the signal. The gradient is right; it is the finite difference that is noisy.
- At **64 days** backprop breaks: it jumps between −0.05 and −0.38 K per unit albedo while the chords sit at −0.14. The breakdown lies somewhere between 32 and 64 days.

> A global mean averages out the weather. Its gradient is reliable for weeks, which makes restoring a global mean the easy control problem.
{: .block-tip }

## Bad gradients: land temperature error

{% include theme_figure.liquid dir="grad_web" stem="fig2_t2m_land_mse_slice" title="Land T2m error vs ocean cloud albedo" %}
<div class="caption">
    Mean-square error of the land T2m map against the 1×CO₂ control run, 4×CO₂ climate, 5-member mean. Same layout.
</div>

- $$J(s)$$ is already jagged at **4 days**: the land temperature map carries weather noise that a 0.025 step cannot beat.
- Chord / backprop is 0.92 and 0.82 at 4 and 8 days, and 1.4–1.7 at 16–32 days, where the narrow and wide chords also disagree with each other.
- At 64 days backprop is off by more than an order of magnitude (−24.5 vs −1.6 K² per unit albedo).

## Precipitation: almost no usable signal

{% include theme_figure.liquid dir="grad_web" stem="fig3_pr_land_mse_slice" title="Land precipitation error vs ocean cloud albedo" %}
<div class="caption">
    Mean-square error of the land precipitation map against the 1×CO₂ control run, 4×CO₂ climate, 5-member mean. Same layout.
</div>

- Backprop and wide chords agree in sign through 32 days but not in size (chord / backprop 1.2–3.4).
- Over the full sweep the error changes by **under 1%** of its value: brightening ocean clouds barely moves land precipitation within weeks, while precipitation itself is noisy.

This matches what we saw when [optimizing precipitation directly](/blog/2026/receding-horizon-climate-control/#new-targeting-precipitation-directly-fails): the optimizer made precipitation worse.

## Signal to noise picks usable pairs

Agreement with finite differences says whether a gradient is *right*. Whether it is *useful* depends on how much a step along it changes the objective compared with the noise. We define the **step SNR**

$$
\text{SNR} = \frac{\lvert dJ/ds \rvert \, \Delta s}{\sigma_J},
$$

where $$\Delta s$$ is a step in albedo and $$\sigma_J$$ is the jitter of $$J(s)$$ about a straight-line fit across the sweep, the noise one run sees. SNR > 1 means one step predicts a change larger than the noise.

{% include theme_figure.liquid dir="grad_web" stem="fig4_step_snr" title="Step signal-to-noise ratio vs horizon" %}
<div class="caption">
    Step SNR for Δs = 0.1 against horizon H. Dots: members; lines: member median. Dotted: SNR = 1. At 64 days backprop disagrees with FD, so the SNR there is not real.
</div>

Member-median step SNR at $$\Delta s = 0.1$$:

| Horizon | Global-mean SST | Land T2m error | Land precip error |
|---|---|---|---|
| 4 d | **38** | 2.4 | 0.14 |
| 8 d | **57** | 2.1 | 0.28 |
| 16 d | **56** | 0.17 | 0.27 |
| 32 d | **15** | 0.14 | 0.15 |
{: .table .table-sm .table-hover }

- **Global-mean SST** stays far above 1 through 32 days: every member, every horizon.
- **Land T2m error** is just above 1 at 4–8 days and drops tenfold below 1 from 16 days on.
- **Land precipitation error** stays below 1 at every horizon: one albedo step of 0.1 predicts a change smaller than the noise of a single run.
- The 64-day values are not real: there backprop disagrees with the finite differences for all three targets.

## Takeaways

- **Check the gradient before optimizing.** A sweep along one direction with FD chords is cheap and shows where backprop holds.
- **Global means are easy targets.** Their gradients hold for weeks.
- **Regional and pattern targets are noisy.** Land temperature error is noisy within days; land precipitation error barely responds.
- **Step SNR ranks actuator-target pairs.** It folds the gradient size, the step and the weather noise into one number.
