---
layout: page
title: teaching
permalink: /teaching/
description: Courses I've taugh as a TA.
nav: true
nav_order: 3
display_categories: [mechanics, computing]
horizontal: false
---

<!-- pages/teaching.md -->
<div class="teaching">
{% if site.enable_teaching_categories and page.display_categories %}
  <!-- Display categorized teaching resources -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_teaching = site.teaching | where: "category", category %}
  {% assign sorted_teaching = categorized_teaching | sort: "importance" %}
  <!-- Generate cards for each teaching resource -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-2">
    {% for resource in sorted_teaching %}
      {% include teaching_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="grid">
    {% for resource in sorted_teaching %}
      {% include teaching.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display teaching resources without categories -->

{% assign sorted_teaching = site.teaching | sort: "importance" %}

  <!-- Generate cards for each teaching resource -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-2">
    {% for resource in sorted_teaching %}
      {% include teaching_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="grid">
    {% for resource in sorted_teaching %}
      {% include teaching.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>
