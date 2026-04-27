---
layout: page
title: Climate Emulator Comparison
description: How three ML-based atmosphere emulators respond to uniform ocean warming
img: assets/img/emulator_comparison_t2m.png
importance: 5
category:
related_publications: false
---

We ran three machine-learning atmosphere emulators under a +4 K uniform sea surface temperature (SST) perturbation and compared how the near-surface air temperature responds across the globe.

The three emulators are [LUCIE](https://arxiv.org/abs/2509.02061), an SFNO-based model trained on ERA5 reanalysis data; [ACE2-ERA5](https://arxiv.org/abs/2411.11268), a transformer-based model trained on ERA5; and [NeuralGCM](https://www.nature.com/articles/s41586-024-07744-y), a hybrid dynamical-core / learned-physics model based on ERA5. While the global temperature fields in the +4K run appear similar across emulators, the differences become prominent when we subtract the control from the warmed fields.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/emulator_comparison_t2m.png" title="3-way T2m difference: LUCIE / ACE2-ERA5 / NeuralGCM" class="img-fluid rounded z-depth-1 w-100" %}
    </div>
</div>
<div class="caption text-center">
    Left column: time-averaged near-surface temperature in the +4K run. Right column: difference (+4K minus control). LUCIE uses its dedicated T2m diagnostic; ACE2-ERA5 uses its 2m air temperature output; NeuralGCM uses 1000 hPa temperature (no T2m diagnostic available). LUCIE and ACE2-ERA5 are averaged over a 5-year run; NeuralGCM over the first stable year. Global-mean ΔT is annotated on each difference panel.
</div>

Despite being forced identically, the response of the three emulators differs significantly.

- **LUCIE** (+2.79 K global mean): moderate, spatially smooth warming with realistic polar amplification.
- **ACE2-ERA5** (+0.82 K global mean): weaker overall warming; notably, several land regions show *cooling* relative to control. This is a physically unexpected response under uniform SST forcing.
- **NeuralGCM** (+3.66 K global mean): the strongest response with polar amplification and physically plausible high temperatures on land. Closest to the expected response but utility is limited due to instability over long runs.

These differences reflect distinct learned climate sensitivities baked into each model's training objective and architecture. Characterizing these sensitivities is crucial for taking advantage of these differentiable surrogate models. Our ongoing work dives deeper into the sensitivities of these emulators.
