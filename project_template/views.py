from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from .test import find_similar
from .tm import baseIR#, TMRetrieval
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# Create your views here.
def index(request):
    output_list = ''
    output= ''
    search = ''
    if request.GET.get('search'):
        search = request.GET.get('search')
        output_list = baseIR(search)
        # output_list = TMRetrieval([search], 20)
        paginator = Paginator(output_list, 5)
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