---
layout: page
title: Archive
permalink: /archive/
---

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}