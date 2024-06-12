from django.forms import ModelForm
from django import forms
from .models import *


# 1. Типы отчетов
class ReportTypeForm(forms.ModelForm):
    class Meta:
        model = ReportType
        fields = '__all__'


# 2. Источник
class SourceForm(forms.ModelForm):
    class Meta:
        model = Source
        fields = '__all__'


# 3. Регион
class RegionForm(forms.ModelForm):
    class Meta:
        model = Region
        fields = '__all__'

# 4. Отчет
class ReportForm(forms.ModelForm):
    class Meta:
        model = Report
        fields = '__all__'

