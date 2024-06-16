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


# 1. Типы отчетов
class ReportType(models.Model):
    # Название типа отчетов
    report_type = models.CharField(max_length=20, help_text='')

    def __str__(self):
        return self.report_type

    def get_absolute_url(self):
        return reverse('report_type-detail', args=[str(self.id)])


# 2. Источник
class Source(models.Model):
    # Название источника
    source = models.CharField(max_length=40, help_text='')
    # URL источника
    URL = models.CharField(max_length=40, help_text='')

    def __str__(self):
        return self.source

    def get_absolute_url(self):
        return reverse('source-detail', args=[str(self.id)])


# 3. Регион
class Region(models.Model):
    # Название месторождения
    region = models.CharField(max_length=20, help_text='')
    # Особенности региона
    region_feature = models.CharField(max_length=200, help_text='')

    def __str__(self):
        return self.region

    def get_absolute_url(self):
        return reverse('region-detail', args=[str(self.id)])


# 4. Отчет
class Report(models.Model):
    # Название отчета
    report_name = models.CharField(max_length=200, null=True, help_text='')
    # Краткое содержание отчета
    short_description = models.CharField(max_length=200, null=True, help_text='')
    # Тип отчета
    report_type = models.ForeignKey(ReportType, on_delete=models.SET_NULL, null=True, help_text='')
    # Источники
    sources = models.ManyToManyField(Source, null=True, help_text='')
    # Краткое содержание отчета
    additional_sources = models.CharField(max_length=200, null=True, help_text='')
    # География
    regions = models.ManyToManyField(Region, null=True, help_text='')
    # Фильтры
    filters = models.CharField(max_length=200, null=True, help_text='')
    # Данные для графиков
    plot_files = models.FileField(upload_to='documents/graphics', null=True, blank=True)
    # Дополнительные файлы
    files = models.FileField(upload_to='documents/files', null=True, blank=True)
    # Необходимые главы
    chapters = models.CharField(max_length=200, null=True, help_text='')
    # Графики
    graphics = models.FileField(upload_to='documents/graphics', null=True, blank=True)
    # Название сгеренерированной главы 1
    chapter_name_1 = models.CharField(max_length=200, null=True, help_text='')
    # Текст сгеренерированной главы 1
    chapter_text_1 = models.CharField(max_length=2000, null=True, help_text='')
    # Название сгеренерированной главы 2
    chapter_name_2 = models.CharField(max_length=200, null=True, help_text='')
    # Текст сгеренерированной главы 2
    chapter_text_2 = models.CharField(max_length=2000, null=True, help_text='')
    # Название сгеренерированной главы 3
    chapter_name_3 = models.CharField(max_length=200, null=True, help_text='')
    # Текст сгеренерированной главы 3
    chapter_text_3 = models.CharField(max_length=2000, null=True, help_text='')
    # Название сгеренерированной главы 4
    chapter_name_4 = models.CharField(max_length=200, null=True, help_text='')
    # Текст сгеренерированной главы 4
    chapter_text_4 = models.CharField(max_length=2000, null=True, help_text='')
    # Название сгеренерированной главы 5
    chapter_name_5 = models.CharField(max_length=200, null=True, help_text='')
    # Текст сгеренерированной главы 5
    chapter_text_5 = models.CharField(max_length=2000, null=True, help_text='')
    # Название сгеренерированной главы 6
    chapter_name_6 = models.CharField(max_length=200, null=True, help_text='')
    # Текст сгеренерированной главы 6
    chapter_text_6 = models.CharField(max_length=2000, null=True, help_text='')

    def get_absolute_url(self):
        return reverse('report-detail', args=[str(self.id)])

    def __str__(self):
        return self.report_name
