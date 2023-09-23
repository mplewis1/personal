---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

{% assign recent_post = site.posts | first %}
<h2>{{ recent_post.title }}</h2>
<p>{{ recent_post.content }}</p>