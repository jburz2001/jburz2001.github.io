---
layout: page
title: courses
permalink: /courses/
description: Courses that Justin Burzachiello has worked on
nav: true
display_categories: [Computer Science, Mathematics, Physics, Chemistry, Social Science, Humanities]
horizontal: false
---
<div class="courses">
  {% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized courses -->
    {% for category in page.display_categories %}
      <h2 class="category">{{ category }}</h2>
      {% assign categorized_courses = site.courses | where: "category", category %}
      {% assign sorted_courses = categorized_courses | sort: "importance" %}
      <!-- Generate cards for each project -->
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-2">
          {% for project in sorted_courses %}
            {% include courses_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for project in sorted_courses %}
            {% include courses.html %}
          {% endfor %}
        </div>
      {% endif %}
    {% endfor %}

  {% else %}
  <!-- Display courses without categories -->
    {% assign sorted_courses = site.courses | sort: "importance" %}
    <!-- Generate cards for each project -->
    {% if page.horizontal %}
      <div class="container">
        <div class="row row-cols-2">
        {% for project in sorted_courses %}
          {% include courses_horizontal.html %}
        {% endfor %}
        </div>
      </div>
    {% else %}
      <div class="grid">
        {% for project in sorted_courses %}
          {% include courses.html %}
        {% endfor %}
      </div>
    {% endif %}

  {% endif %}

</div>
