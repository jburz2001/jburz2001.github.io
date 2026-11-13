---
layout: page
title: videos
permalink: /videos/
description: Technical talks and lectures by Justin.
nav: true
nav_order: 3
---

<!-- Videos are listed in _data/videos.yml. Add a new entry there; this page does not need to change. -->

{% assign sections = site.data.videos | where_exp: "s", "s.videos.size > 0" %}

{% if sections.size == 0 %}

<p class="videos-empty">New videos coming soon.</p>
{% else %}
<div class="videos">
  {% if sections.size > 1 %}
    <nav class="videos-toc" aria-label="Video topics">
      {% for section in sections %}
        <a href="#{{ section.topic | slugify }}">{{ section.topic }}</a>
      {% endfor %}
    </nav>
  {% endif %}

{% for section in sections %}

<section class="videos-section" id="{{ section.topic | slugify }}">
<h2 class="videos-heading">{{ section.topic }}</h2>
{% if section.description %}<p class="videos-section-description">{{ section.description }}</p>{% endif %}
<div class="videos-grid">
{% assign section_videos = section.videos | sort: "date" | reverse %}
{% for video in section_videos %}
{% include video_card.liquid video=video %}
{% endfor %}
</div>
</section>
{% endfor %}
</div>
{% endif %}
