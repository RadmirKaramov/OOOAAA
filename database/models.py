from django.db import models
from django.urls import reverse  # Used to generate URLs by reversing the URL patterns
from django.conf import settings
from django.db.models import Q
from django.utils import timezone
import datetime


# Менеджер для поиска в свойствах
class PropertyManager(models.Manager):
    def search(self, query=None):
        qs = self.get_queryset()
        if query is not None:
            or_lookup = (Q(property__icontains=query) |
                         Q(comment__icontains=query)
                         )
            qs = qs.filter(or_lookup).distinct()  # distinct() is often necessary with Q lookups
        return qs


# 2. Источник
class Source(models.Model):
    # Название источника
    source = models.CharField(max_length=40, help_text='')
    # URL источника
    URL = models.CharField(max_length=40, help_text='')
    # Особенности источника
    source_feature = models.CharField(max_length=40, help_text='')

    def __str__(self):
        return self.source

    def get_absolute_url(self):
        return reverse('source-detail', args=[str(self.id)])


# 3. Регион
class Region(models.Model):
    # Название региона
    region = models.CharField(max_length=20, help_text='')
    # Особенности региона
    region_feature = models.CharField(max_length=200, help_text='')

    def __str__(self):
        return self.region

    def get_absolute_url(self):
        return reverse('region-detail', args=[str(self.id)])


# 1. Шаблон отчетов
class ReportType(models.Model):
    # Название шаблона отчетов
    report_type = models.CharField(max_length=50, help_text='')
    # Описание шаблона отчета
    report_type_description = models.CharField(max_length=50, help_text='')
    # Источники
    sources = models.ManyToManyField(Source, help_text='')
    # География
    regions = models.ManyToManyField(Region, help_text='')
    # Необходимые главы
    chapters = models.CharField(max_length=10000, help_text='')
    # Основной промпт генерации
    chapter_prompt = models.CharField(max_length=10000, null=True, help_text='')
    # Промпт для генерации поисковых запросов
    prompt_search = models.CharField(max_length=10000, null=True, help_text='')
    # Промпт для генерации векторных запросов
    prompt_vector = models.CharField(max_length=10000, null=True, help_text='')

    def __str__(self):
        return self.report_type

    def get_absolute_url(self):
        return reverse('report_type-detail', args=[str(self.id)])


# 4. Отчет
class Report(models.Model):
    # Название отчета
    report_name = models.CharField(max_length=200, null=True, help_text='')
    # Краткое содержание отчета
    short_description = models.CharField(max_length=200, null=True, help_text='')
    # Тип отчета
    report_type = models.ForeignKey(ReportType, on_delete=models.SET_NULL, null=True, help_text='')
    # Дополнительные истончики
    additional_sources = models.CharField(max_length=200, null=True, help_text='')
    # Фильтры
    filters = models.CharField(max_length=10000, null=True, help_text='')
    # Начальная дата
    start_date = models.DateField()
    # Конечная дата
    end_date = models.DateField()
    # Данные для графиков
    plot_files = models.FileField(upload_to='documents/graphics', null=True, blank=True)
    # Дополнительные файлы
    files = models.FileField(upload_to='documents/files', null=True, blank=True)


    def get_absolute_url(self):
        return reverse('report-detail', args=[str(self.id)])

    def __str__(self):
        return self.report_name


# 5. Главы (блоки) отчетов
class Chapter(models.Model):
    # Название главы
    chapter_name = models.CharField(max_length=200, help_text='')
    # Текст главы
    chapter_text = models.TextField()
    # Отчет, к которому принадлежит глава
    report = models.ForeignKey(Report, on_delete=models.SET_NULL, null=True, help_text='')
    # Ответ проверки достоверности
    chapter_validation = models.CharField(max_length=2000, help_text='', blank=True, null=True)
    # Параметры анализа
    chapter_analysis_parameters = models.CharField(max_length=2000, help_text='', blank=True, null=True)
    # Пользовательские аннотации
    chapter_comments = models.CharField(max_length=2000, help_text='', blank=True, null=True)
    # Графики
    graphics = models.FileField(upload_to='documents/graphics', null=True, blank=True)

    def __str__(self):
        return self.chapter_name

    def get_absolute_url(self):
        return reverse('chapter-detail', args=[str(self.id)])