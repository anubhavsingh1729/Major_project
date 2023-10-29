from django import forms

class Imageupload(forms.Form):
    image = forms.ImageField()