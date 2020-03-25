from django.http import HttpResponse
from django.shortcuts import render
from .forms import UploadFileForm
from django.http import HttpResponseRedirect
from .handleFile import handle_uploaded_file

def hello(request):
    context={}
    context['hello']='hello world'
    return render(request,'hello.html',context)

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            context = handle_uploaded_file(request.FILES['file'])
            return render(request, 'info.html', context)
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})