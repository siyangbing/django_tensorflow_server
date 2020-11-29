from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def kougai(request):
    return HttpResponse("eval_img")
