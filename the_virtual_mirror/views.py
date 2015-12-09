from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.shortcuts import render


def welcome(request):
    return render_to_response('welcome.html')

