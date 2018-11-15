import os

import jsonify as jsonify
from django.http import HttpResponse
from django.shortcuts import render


i = 1
# Create your views here.
def index(request):
    return render(request, 'index.html')


def get_image(request):
    print(request.POST)
    global i
    i += 1
    image = open('img/CA_'+str(i)+'.png').read()
    return HttpResponse(image, content_type='image/png')