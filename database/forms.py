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
        labels = {
            "report_name": "Название отчета",
            "short_description": "Краткое содержание отчета",
            "report_type": "Тип отчета",
            "additional_sources": "Дополнительные источники",
            "filters": "Фильтры",
            "start_date": "Начальная дата",
            "end_date": "Конечная дата",
            "plot_files": "Данные для графиков",
            "files": "Дополнительные файлы в формате PDF",
        }


    def __init__(self, *args, **kwargs):
        super(ReportForm, self).__init__(*args, **kwargs)
        self.fields['plot_files'].required = False   
        self.fields['files'].required = False

        self.fields['report_name'].widget.attrs['name'] = 'Название отчета'
        self.fields['short_description'].widget.attrs['name'] = 'Краткое содержание отчета'
        self.fields['report_type'].widget.attrs['name'] = 'Тип отчета'
        self.fields['additional_sources'].widget.attrs['name'] = 'Дополнительные источники'
        self.fields['filters'].widget.attrs['name'] = 'Фильтры'
        self.fields['start_date'].widget.attrs['name'] = 'Начальная дата'
        self.fields['end_date'].widget.attrs['name'] = 'Конечная дата'
        self.fields['plot_files'].widget.attrs['name'] = 'Данные для графиков'
        self.fields['files'].widget.attrs['name'] = 'Дополнительные файлы'

# 5. Глава
class ChapterForm(forms.ModelForm):

    class Meta:
        model = Chapter
        fields = '__all__'
