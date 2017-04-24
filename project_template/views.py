from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from .models import Docs
from django.template import loader
from .form import QueryForm
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

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

        if request.GET.get('search'):
            search = request.GET.get('search')

            if version == 'v1':
                output_list = baseIR(search)

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