from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse
import urllib
import wikipedia
from bs4 import BeautifulSoup
from .v3 import relevant_author, related_topics

# Create your views here.
def index(request):
    output_list = ''
    output= ''
    search = ''

    try:
        topics = map(lambda x: int(x), request.GET.getlist('topic[]', None))
    except Exception as e:
        topics = []

    if request.GET.get('version'):
        version = request.GET.get('version')
        print 'using version {}'.format(version)
        if version == 'v1':
            from .v1 import baseIR as retrieval1
        if version == 'v2':
            from .v2 import BTMRetrieval as retrieval2
        if version == 'v3':
            from .v3 import BTMRetrieval as retrieval3

        if request.GET.get('search'):
            search = request.GET.get('search')

            if version == 'v1':
                output_list = retrieval1(search)
            if version == 'v2':
                output_list = retrieval2(search, 100)
            if version == 'v3':
                if topics == []:
                    output_list = retrieval3(search, 100)
                else:
                    print topics
                    output_list = retrieval3(search, 100, filter_by=topics)

            paginator = Paginator(output_list, 15)
            page = request.GET.get('page')
            try:
                output = paginator.page(page)
            except PageNotAnInteger:
                output = paginator.page(1)
            except EmptyPage:
                output = paginator.page(paginator.num_pages)

    return render_to_response('project_template/index.html', 
                          {'input': search,
                           'output': output,
                           'topics': topics,
                           'magic_url': request.get_full_path(),
                           })

def getImgSrc(page):
  soup = BeautifulSoup(page.html())
  table = soup.find('table', class_='infobox')
  try:
    return table.find_all('img')[0].get('src')
  except Exception as e:
    return 

def getWiki(request):
    author = request.GET.get('author', None)
    try:
        page = wikipedia.page(author)
        pageurl = page.url
        extraction = page.summary

    except Exception as e:
        pageurl = 'https://en.wikipedia.org/wiki/Special:Search?search=' + urllib.quote_plus(author) + '&go=Go'
        extraction = 'No information found for '+author+' on Wikipedia.'

    finally:

        try:
            src = getImgSrc(page)
            assert src != None
        except Exception as e:
            src = ''

        return JsonResponse({'pageurl': pageurl, 
            'extraction': extraction, 
            'src': src
            })

def wikiShowMore(request):
    search = request.GET.get('query', None)
    author = request.GET.get('author', None)
    qID = int(request.GET.get('qID', None))

    try:
        authors = relevant_author(search,qID)
    except Exception as e:
        print e
        authors = []

    try:
        topics = related_topics(search, qID)
        print topics
    except Exception as e:
        print e
        topics = []

    return JsonResponse({'topics': topics,
            'authors': authors})
