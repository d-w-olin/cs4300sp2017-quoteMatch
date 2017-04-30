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

# Create your views here.
def index(request):
    output_list = ''
    output= ''
    search = ''

    if request.GET.get('version'):
        version = request.GET.get('version')
        print 'using version {}'.format(version)
        if version == 'v1':
            from .v1 import baseIR
        if version == 'v2':
            from .v2 import BTMRetrieval

        if request.GET.get('search'):
            search = request.GET.get('search')

            if version == 'v1':
                output_list = baseIR(search)
            if version == 'v2':
                output_list = BTMRetrieval(search, 100)

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
                           'magic_url': request.get_full_path(),
                           })

def getImgSrc(page):
  soup = BeautifulSoup(page.html())
  table = soup.find('table', class_='infobox')
  try:
    return table.find_all('img')[0].get('src')
  except Exception as e:
    print e
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
        topics = []
        authors = ['author 1', 'author 2', 'author 3']
        try:
            src = getImgSrc(page)
            print 'src is {}!'.format(src)
            assert src != None
        except Exception as e:
            print e
            src = ''

        return JsonResponse({'pageurl': pageurl, 
            'extraction': extraction, 
            'src': src,
            'topics': topics,
            'authors': authors
            })



















