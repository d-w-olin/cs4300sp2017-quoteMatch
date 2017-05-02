from django.conf.urls import url

from . import views

app_name = 'pt'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^getWiki/$', views.getWiki, name='getWiki'),
    url(r'^wikiShowMore/$', views.wikiShowMore, name='wikiShowMore')
]