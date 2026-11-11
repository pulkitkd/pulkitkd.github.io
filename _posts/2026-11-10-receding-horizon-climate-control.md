---
layout: post
title: Receding Horizon Climate Control
date: 2026-11-10 12:00:00
description: Optimizing ocean cooling patterns through a differentiable climate model
tags: climate geoengineering
categories: research
related_posts: false
toc:
  sidebar: left
---

> **In short.** We optimize a time-varying pattern of sea-surface cooling by backpropagating through a differentiable atmospheric model, a few days at a time. The learned strategy returns land temperature to its unwarmed climate and **holds it for years**, although no gradient spans more than 8 days. When the objective targets the spatial pattern of land temperature, precipitation, evaporation and humidity are restored too, although they never enter the objective. Targeting precipitation directly does worse than doing nothing.
{: .block-tip }

## The setup

Proposals to offset global warming, such as marine cloud brightening and stratospheric aerosol injection, pose an inverse design problem: **which pattern of forcing, in space and time, best restores a target climate?** Existing approaches linearize the climate response, fix the forcing to a few prescribed shapes, or search by trial. A differentiable climate model allows a more direct route, gradient-based optimization through the model itself.

- **Model.** JAX-GCM, a differentiable atmospheric model: the spectral primitive-equation core of NeuralGCM coupled to SPEEDY physics.
- **Perturbation.** A uniform **+4 K** sea-surface temperature (SST) warming.
- **Control.** Five ocean-only Gaussian zonal bands of SST perturbation, centered at 45°N, 20°N, 0°, 20°S and 45°S, each 10° wide. Their amplitudes $$a_1(t), \dots, a_5(t)$$ are free.
- **Target.** Near-surface air temperature (T2m) over land, returned to the model's own unwarmed climatology.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/rhc/fig1_schematic.png" title="Problem setup and optimization scheme" class="img-fluid rounded z-depth-1 w-100" %}
    </div>
</div>
<div class="caption">
    (A) A uniform +4 K SST offset warms the climate; five zonal bands with free amplitudes supply the corrective forcing. (B) Greedy segmented optimization. (C) Time-mean forcing learned under the pattern and land-mean objectives. (D) Zonal-mean precipitation residual for the two objectives. (E) The learned strategy, replayed without re-optimization in LUCIE and NeuralGCM.
</div>

## The method

### Why short horizons

A gradient backpropagated through chaotic dynamics grows like $$e^{\Lambda T}$$, where $$\Lambda$$ is the leading Lyapunov exponent, even though the true sensitivity of a time-averaged objective stays bounded. Beyond a few Lyapunov times (about **one week** for the mid-latitude atmosphere), a single-trajectory gradient carries no usable signal. In our tests the gradients degrade beyond a 14-day horizon and fail by about 28–30 days.

### Receding horizon optimization

We therefore optimize greedily, following the receding-horizon approach of adjoint-based turbulence control:

1. Split the run into **8-day segments**; within a segment the five amplitudes are held constant.
2. For the current segment, find the amplitudes that minimize the loss by exact backpropagation through the 8-day rollout.
3. Apply them, advance the model to the end of the segment, and repeat.

No gradient crosses a segment boundary. Each segment minimizes

$$
L = \alpha \langle d \rangle_w^2 + \beta \left\langle (d - \langle d \rangle_w)^2 \right\rangle_w + \mu\, \overline{a_j^2} + \lambda\, \overline{(a_j - a_j^-)^2},
$$

where $$d(\mathbf{x}) = T_\text{opt}(\mathbf{x}) - T_\text{target}(\mathbf{x})$$ is the segment-mean land T2m residual, $$\langle \cdot \rangle_w$$ is the area-weighted land mean, and the overbar averages over the five bands.

- **Land-mean bias** ($$\alpha$$): how far the land-average temperature is from the target.
- **Pattern** ($$\beta$$): the spatial variance of the residual. The *land-mean objective* sets $$\beta = 0$$; the *pattern objective* sets $$\alpha = 1$$, $$\beta = 0.5$$.
- **Regularization** ($$\mu$$, $$\lambda$$): penalizes large amplitudes and their change from the previous segment's $$a_j^-$$.

## Result: two years of control from 8-day windows

The optimized run reaches the target within about **10 segments (10 weeks)**, then tracks the unwarmed seasonal cycle for two years. Ensemble members, started from different initial amplitudes, differ only in those first 10 weeks. A 14-day-segment run over three years converges to a similar strategy.

{% include theme_figure.liquid dir="rhc_web" stem="fig2a_timeseries" title="Land-mean T2m over two years" %}
<div class="caption">
    Pattern objective, 91 segments × 8 days, 10 members (mean ± 1σ; 40-day moving average). Land-mean T2m for the warmed (+4 K), target and optimized runs; a dot marks every fifth segment.
</div>

{% include theme_figure.liquid dir="rhc_web" stem="fig2bcd_bands" title="Zonal T2m and band amplitudes" %}
<div class="caption">
    (b) Zonal-mean T2m. (c) Each band's amplitude as a deviation from its time mean, drawn about the band's own latitude (scale bar ±1 K). (d) Time-mean band amplitudes (filled) and the ocean-fraction and area-weighted applied forcing (line).
</div>

## Result: the pattern objective restores the water cycle

Both objectives remove a comparable fraction of the land warming, but they place the cooling at different latitudes.

- **Land-mean objective.** Concentrates cooling on the equatorial band, which carries **34%** of the applied forcing. That band sits under the ITCZ; the cooling suppresses convection and **dries the equator**.
- **Pattern objective.** Applies shallower equatorial cooling and spreads the rest across the bands. The zonal-mean residuals of precipitation, evaporation and humidity stay **near zero**; the equatorial precipitation residual is almost **10× smaller**.

{% include theme_figure.liquid dir="rhc_web" stem="fig3a_forcing" title="Applied forcing" %}
<div class="caption">
    Time-mean applied forcing over the final year, pattern (blue) vs land-mean (crimson) objective.
</div>

{% include theme_figure.liquid dir="rhc_web" stem="fig3b_hydro" title="Hydrological response" %}
<div class="caption">
    Zonal-mean precipitation, evaporation and specific humidity (top) and their departures from the unwarmed target (bottom; +4 K departure shaded). None of these variables is in the loss.
</div>

## Result: restoration scores

We score restoration with the **gain** $$G$$: the mean-square error of the uncontrolled +4 K run against the target, divided by that of the controlled run, on time-mean land fields over the final year. $$G = 10$$ means the error is one tenth of doing nothing; $$G = 1$$ means no improvement.

At the same effort as a flat cooling on all bands (**3.44 K**), the pattern strategy wins on every variable:

| Variable | Learned (pattern) | Matched-effort flat |
|---|---|---|
| T2m | **27.4** | 20.8 |
| Evaporation | **49.2** | 23.6 |
| Specific humidity | **74** | 34 |
| Precipitation | **9.8 ± 1.7** | 6.9 ± 0.7 |
{: .table .table-sm .table-hover }

{% include theme_figure.liquid dir="rhc_web" stem="fig4_gain_effort" title="Gain vs effort" %}
<div class="caption">
    Gain G against effort E (land, final year) for the learned strategies and four baselines (matched effort, matched pattern, flat −4 K, linear response). The dashed line joins the highest-gain strategy at each effort. Matched pattern sits just below Learned (pattern) at the same effort.
</div>

## New: targeting precipitation directly fails

If a temperature objective restores precipitation as a side effect, would targeting precipitation directly do better? We reran the optimizer with the loss replaced by the **mean-square error of the land precipitation field** against the unwarmed target, with no temperature term. Each run is compared with a flat cooling of equal effort on all five bands (5 members).

Skill here is $$R = 1 - \mathrm{RMS}(\text{controlled} - \text{target}) / \mathrm{RMS}(+4\,\text{K} - \text{target})$$ on time-mean fields over the final year. $$R = 0$$ is no improvement over the +4 K run; **$$R < 0$$ is worse than doing nothing**.

- **With amplitude penalties** (1 run, effort 1.9 K): land precipitation **R = −0.19**, vs **0.52 ± 0.04** for flat cooling. Land T2m 0.19 vs 0.52.
- **Without penalties** (3 runs, effort 5.9 K): land precipitation **R = −0.05 ± 0.06** vs 0.04 ± 0.04; global precipitation **−0.38 ± 0.07** vs 0.12 ± 0.05.

> In both cases the precipitation-optimized run leaves precipitation **worse than the uncontrolled warmed climate** and loses to flat cooling of equal effort. The temperature-pattern objective, which never sees precipitation, restores it far better ($$G = 9.8$$).
{: .block-warning }

**With penalties** (1 run):

{% include theme_figure.liquid dir="rhc_web" stem="prland_pen_timeseries" title="Precipitation loss with penalties: land-mean T2m" %}
{% include theme_figure.liquid dir="rhc_web" stem="prland_pen_bands" title="Precipitation loss with penalties: bands" %}
<div class="caption">
    Same layout as the two-year control figure, for the precipitation-loss run with amplitude penalties.
</div>

**Without penalties** (3 runs):

{% include theme_figure.liquid dir="rhc_web" stem="prland_pen0_timeseries" title="Precipitation loss without penalties: land-mean T2m" %}
{% include theme_figure.liquid dir="rhc_web" stem="prland_pen0_bands" title="Precipitation loss without penalties: bands" %}
<div class="caption">
    Same layout, precipitation loss without penalties (mean ± 1σ, n = 3). The 45°N band is driven far beyond the ±1 K lane scale (time mean about −21 K), so the lanes in (c) sit at their clip limit; note the wider forcing axis in (d).
</div>

**Skill against flat cooling of equal effort:**

{% include theme_figure.liquid dir="rhc_web" stem="prland_pen_skill" title="Skill, with penalties" %}
{% include theme_figure.liquid dir="rhc_web" stem="prland_pen0_skill" title="Skill, without penalties" %}
<div class="caption">
    Restoration skill R of the precipitation-optimized runs (green) vs matched-effort flat cooling (grey, n = 5, ±1σ), global and land, final year. Top: with penalties. Bottom: without. Dotted ticks mark the ceiling.
</div>

## Takeaways

- **Short-horizon gradients give long-horizon control.** No gradient spans more than 8 days, yet the stitched strategy holds the target for years.
- **The objective shapes the side effects.** Matching the temperature *pattern*, not just its mean, restores the water cycle.
- **Where the cooling goes matters, not just how much.** At equal effort the learned pattern beats flat cooling on every variable; the gain comes from the spatial pattern, not the time variation.
- **Not every variable is a good target.** Optimizing precipitation directly made it worse.
- **The method is general.** It applies to any differentiable model and objective; the JAX-GCM strategy, replayed in LUCIE and NeuralGCM without re-optimization, tracks each model's target.
