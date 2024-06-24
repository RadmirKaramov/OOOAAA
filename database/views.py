from django.shortcuts import render
from django.views import generic
from .models import *
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from .uploads import handle_uploaded_tests, handle_uploaded_properties
from .forms import *
from itertools import chain
from django.views.generic import ListView
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from .filters import ReportsFilter
from django_filters.views import FilterView
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
import ast
from langchain_community.document_loaders import pdf
import matplotlib.pyplot as plt
from django.core.files import File

import gc
from docx import Document
import pypandoc

import io

import csv

### AI stuff on
import uuid

import asyncio
import langchain_community
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re

from openai import OpenAI
import httpx

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Apply nest_asyncio to allow nested event loops
import nest_asyncio
nest_asyncio.apply()

import time
import os
from dotenv import load_dotenv
import json

load_dotenv()

OPEN_AI_API_TOKEN = os.getenv('OPEN_AI_API_TOKEN')
PROXY = os.getenv('PROXY')

client = OpenAI(api_key = OPEN_AI_API_TOKEN,
                     http_client=httpx.Client(
                         proxies=PROXY,
                         transport = httpx.HTTPTransport(local_address='0.0.0.0')
                     ))

user_agent = "(Standard) Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 YaBrowser/24.4.4.1160 Yowser/2.5 Safari/537.36"

def ddg_search(json_string, user_agent, sources):
    content = []
    data = json.loads(json_string)
    queries = [data[key] for key in data]

    for query in queries:
        # Выполняем поиск без указания источников
        results = DDGS().text(query, max_results = 3) # 5 ссылок без источников
        urls = [result['href'] for result in results]

        docs = get_page(urls,user_agent)

        for doc in docs:
            page_text = re.sub("\n\n+", "\n", doc.page_content)
            content.append(doc)

        # Выполняем поиск с указанием источников
        for source in sources:
            query_with_source = f"{query} site:{source}"
            results = DDGS().text(query_with_source, max_results = 2) #3 ссылки для источников
            urls = [result['href'] for result in results]

            docs = get_page(urls,user_agent)

            for doc in docs:
                page_text = re.sub("\n\n+", "\n", doc.page_content)
                content.append(doc)
                
    return content

# retrieves pages and extracts text by tag
def get_page(urls, user_agent):

    loader = AsyncChromiumLoader(urls)
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"], remove_unwanted_tags=["a"])

    return docs_transformed


def format_docs(docs):
    context_prompt = ''
    for doc in docs:
        context_prompt += "\n----------------------------------------------------------------\n\n"
        context_prompt += doc.page_content
        context_prompt += '\n{ источник: ' + doc.metadata['source'] + ' }'
    return context_prompt


def text_retrive(json_string, retriever):
    contexts_list = []
    data = json.loads(json_string)
    queries = [data[key] for key in data]

    for query in queries:
        # Выполняем поиск без указания источников
        retrived_docs = retriever.invoke(query)
        for retrived_doc in retrived_docs:
            contexts_list.append(retrived_doc)

    contexts_list_wo_dup = []
    for obj in contexts_list:
        if obj not in contexts_list_wo_dup:
            contexts_list_wo_dup.append(obj)

    context = format_docs(contexts_list_wo_dup)
    
    return context


def create_completion_openai(client, prompt):
    answer = client.chat.completions.create(
        messages = [{"role":"user",
                     "content": prompt}],
        model = "gpt-4o"
    )
    return answer.choices[0].message.content


# def create_completion_openai(client, prompt):
#     res = client.completions.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         temperature=0,
#         max_tokens=1000,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None
#     )

#     return res.choices[0].text


def create_prompt_search(prompt_search, short_description, start_date, end_date, regions, filters):
    template = prompt_search
    prompt = template.format(short_description=short_description,\
                             start_date=start_date,\
                             end_date=end_date,\
                             regions=regions,\
                             filters=filters)
    return prompt


def create_prompt_vector(prompt_vector, chapter, short_description, start_date, end_date, regions, filters):
    template = prompt_vector
    prompt = template.format(chapter=chapter,\
                             short_description=short_description,\
                             start_date=start_date,\
                             end_date=end_date,\
                             regions=regions,\
                             filters=filters)
    return prompt


def gen_chapter(request, object_id):

    # Подтягивание данных из БД
    report = Report.objects.get(pk=object_id)
    report_name = report.report_name
    short_description = report.short_description
    start_date = report.start_date
    end_date = report.end_date
    add_sources = report.additional_sources

    report_type = report.report_type
    prompt_search = report_type.prompt_search
    prompt_vector = report_type.prompt_vector
    chapter_prompt = report_type.chapter_prompt
    regions_all = report_type.regions.all()
    sources_all = report_type.sources.all()
    chapters_str = report_type.chapters
    chapter_names = ast.literal_eval(chapters_str)
    filters = report.filters

    add_sources_list = add_sources.split(',')

    sources = []

    for source in sources_all:
        sources.append(source.URL)

    for add_source in add_sources_list:
        sources.append(add_source)

    regions = ''
    for region in regions_all:
        regions += region.region+', '

    ## Сбор данных
    # Создание промптов для поиска в инете
    prompts_for_search = create_prompt_search(prompt_search, short_description, start_date, end_date, regions, filters)
    json_prompts = create_completion_openai(client, prompts_for_search)

    # report.filters = report.filters + ' FOR SEARCH ' + json_prompts
    # report.save()

    # Поиск в инете
    docs = ddg_search(json_prompts, user_agent, sources)

    # Добавление пдфок
    pdf_path = report.files
    pdf_doc = pdf.PyPDFLoader(pdf_path).load()
    docs.append(pdf_doc)

    # Проверка дубликатов URL
    docs_wo_duplicates = []
    metadata_list = []

    for obj in docs:
        if obj.metadata['source'] not in metadata_list:
            metadata_list.append(obj.metadata['source'])
            docs_wo_duplicates.append(obj)

    # Создание небольших кусков текста для загрузки в векторную БД
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs_wo_duplicates)

    # Проверка кусков текста на дубликаты
    splits_wo_duplicates = []
    for obj in splits:
        if obj not in splits_wo_duplicates:
            splits_wo_duplicates.append(obj)

    # Загрузка кусков текста в векторную БД и инициализация извлечения
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    fetch_k = len(splits)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 5, 'fetch_k': fetch_k})

    ## Создание главы
    previous_chapters = ''
    for chapter_name in chapter_names:
        # Генерация запросов в векторную БД
        prompts_for_vector = create_prompt_vector(prompt_vector, chapter_name, short_description, start_date, end_date, regions, filters)
        json_v_prompts = create_completion_openai(client, prompts_for_search)


        # Создание контекста для LLM извлечение и подготовкой данных
        context = text_retrive(json_v_prompts, retriever)
        
        # Создание промпта для генерации главы
        prompt = chapter_prompt.format(report_name=report_name,\
                                       short_description=short_description,\
                                       start_date=start_date,\
                                       end_date=end_date,\
                                       regions=regions,\
                                       context=context,\
                                       previous_chapters=previous_chapters,\
                                       chapter_name=chapter_name)
        
        chapter_text = create_completion_openai(client, prompt)
        # chapter_text = 'Какой-то сгенеренный текст в общем'

        # Создание главы, как нового объекта
        new_chapter = Chapter(chapter_name=chapter_name, chapter_text=chapter_text, report=report,\
                           chapter_validation='Пока не проходило проверку достоверности')
        new_chapter.save()

        previous_chapters += chapter_text 
        
        
    # Очистка БД от данных для предотвращения memory leak
    vectorstore.delete(ids=vectorstore.get()['ids'])

    del vectorstore
    gc.collect()

    return HttpResponseRedirect(reverse('report-detail', args=(object_id,)))


def regen_chapter(request, object_id):

    # Подтягивание данных из БД
    chapter = Chapter.objects.get(pk=object_id)
    chapter_parameters = chapter.chapter_analysis_parameters
    chapter_comments = chapter.chapter_comments
    chapter_name = chapter.chapter_name

    report = chapter.report
    report_name = report.report_name
    short_description = report.short_description
    start_date = report.start_date
    end_date = report.end_date

    report_type = report.report_type
    prompt_search = report_type.prompt_search
    prompt_vector = report_type.prompt_vector
    chapter_prompt = report_type.chapter_prompt
    regions_all = report_type.regions.all()
    sources_all = report_type.sources.all()
    chapters_str = report_type.chapters
    chapter_names = chapters_str.split(',')
    filters = report.filters

    all_chapters = ''

    br = '''\\
    \\
    \\'''
    
    chapter_list = Chapter.objects.filter(report = report).order_by('id')
    for chapter_ in chapter_list:
        if object_id == chapter_.id:
            pass
        else:
            all_chapters = all_chapters + br + chapter_.chapter_text

    sources = []
    for source in sources_all:
        sources.append(source.URL)

    regions = ''
    for region in regions_all:
        regions += region.region+', '

    ## Сбор данных
    # Создание промптов для поиска в инете
    prompts_for_search = create_prompt_search(prompt_search, short_description, start_date, end_date, regions, filters)
    json_prompts = create_completion_openai(client, prompts_for_search)

    # report.filters = report.filters + ' FOR SEARCH ' + json_prompts
    # report.save()

    # Поиск в инете
    docs = ddg_search(json_prompts, user_agent, sources)

    # Проверка дубликатов URL
    docs_wo_duplicates = []
    metadata_list = []

    for obj in docs:
        if obj.metadata['source'] not in metadata_list:
            metadata_list.append(obj.metadata['source'])
            docs_wo_duplicates.append(obj)

    # Создание небольших кусков текста для загрузки в векторную БД
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs_wo_duplicates)

    # Проверка кусков текста на дубликаты
    splits_wo_duplicates = []
    for obj in splits:
        if obj not in splits_wo_duplicates:
            splits_wo_duplicates.append(obj)

    # Загрузка кусков текста в векторную БД и инициализация извлечения
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    fetch_k = len(splits)
    vectorstore = Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 5, 'fetch_k': fetch_k})

    ## Создание главы
    previous_chapters = all_chapters

    # Генерация запросов в векторную БД
    prompts_for_vector = create_prompt_vector(prompt_vector, chapter_name, short_description, start_date, end_date, regions, filters)
    json_v_prompts = create_completion_openai(client, prompts_for_search)


    # Создание контекста для LLM извлечение и подготовкой данных
    context = text_retrive(json_v_prompts, retriever)
    
    # Создание промпта для генерации главы
    prompt = chapter_prompt.format(report_name=report_name,\
                                   short_description=short_description,\
                                   start_date=start_date,\
                                   end_date=end_date,\
                                   regions=regions,\
                                   context=context,\
                                   previous_chapters=previous_chapters,\
                                   chapter_name=chapter_name)

    full_prompt =  + f'{prompt}.\n\n Выше приведены контекст, тексты остальных глав отчета.\n\n\
             При создании главы учитывай следующие комментарии к главе: {chapter_comments} \
             \n\n Также учитывай следующие параметры анализа в данной главе: {chapter_parameters}'

    # chapter_text = create_completion_openai(client, full_prompt)

    # Создание главы заново
    chapter.chapter_text = create_completion_openai(client, full_prompt)
    chapter.save()
    # Очистка БД от данных для предотвращения memory leak
    vectorstore.delete(ids=vectorstore.get()['ids'])

    del vectorstore
    gc.collect()

    return HttpResponseRedirect(reverse('chapter-detail', args=(object_id,)))


def create_table(request, object_id):

    # Подтягивание данных из БД
    chapter = Chapter.objects.get(pk=object_id)

    chapter_text = chapter.chapter_text
    table_comments = chapter.table_comments


    prompt = 'Представь текста главы, который представлен выше, в виде таблицы. \
              Выводи только таблицу без лишнего текста.'

    full_prompt = f'{chapter_text} /n/n/n {prompt} /n Также учитывай следующие комментарии: /n {table_comments}'

    # table_text = create_completion_openai(client, full_prompt)

    # Создание главы заново
    chapter.table_text = create_completion_openai(client, full_prompt)
    chapter.save()

    return HttpResponseRedirect(reverse('chapter-detail', args=(object_id,)))


def create_plot(request, object_id):

    # Подтягивание данных из БД
    chapter = Chapter.objects.get(pk=object_id)

    chapter_text = chapter.chapter_text
    plot_comments = chapter.plot_comments


    prompt = 'Предоставь данные, приведенные в главе аналитического отчета (выше) в виде графика.\
          Подписи в графике должны быть на русском языке. \
          Выбирай график (или графики), который считаешь наиболее подходящим. DPI обязательно 70. \
          Выводи только python code с использование plt.show()\
          Ответ должен быть без лишнего текста, не используй ```, выводи только чистый код.'

    full_prompt = f'{chapter_text} /n/n/n {prompt} /n Также учитывай следующие комментарии: /n {plot_comments}'

    plot_code = create_completion_openai(client, full_prompt)

    # Создание и сохранение изобрежения графика в файл
    prompt_plot_name = f'{plot_code} /n/n/n Выше представлен код генерации к главе аналитического отчета \
                         /n Придумай точное, лаконичное название, которое описывает этот график, \
                         /n учитывая текст главы: /n  {chapter_text}. \
                         /n/n/nВыведи только название графика, без ничего лишнего, без #.'

    success = False
    max_attempts = 5
    attempts = 0
    
    while not success and attempts < max_attempts:
        attempts += 1

        try:

            exec(plot_code)
            file_path = f'./documents/graphics/buffer{object_id}.png'
            plt.savefig(file_path)

            with open(file_path, 'rb') as f:

                # Create a Django File instance from the opened file
                django_file = File(f)

                # Assign the File instance to the file field and save
                chapter.graphics.save(file_path, django_file, save=True)
                chapter.save()

                if os.path.exists(file_path):
                    success = True
                else:
                    raise ValueError("Generated code did not produce a graph.")

        except Exception as e:
            print(f"Attempt {attempts} failed: {e}")
    plot_name = create_completion_openai(client, prompt_plot_name)
    chapter.plot_code = plot_code
    chapter.plot_name = plot_name
    chapter.save()

    return HttpResponseRedirect(reverse('chapter-detail', args=(object_id,)))

### AI stuff off


### Django stuff on


@login_required(login_url='/accounts/login/')
def index(request):
    """View function for home page of site."""

    # Generate counts of some of the main objects
    num_visits = request.session.get('num_visits', 0)
    num_report = Report.objects.all().count()
    request.session['num_visits'] = num_visits + 1

    context = {
        'num_report': num_report,
        'num_visits': num_visits,
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)


# 1. Организации
class ReportTypeListView(LoginRequiredMixin, generic.ListView):
    model = ReportType


class ReportTypeDetailView(LoginRequiredMixin, generic.DetailView):
    model = ReportType


class ReportTypeCreate(LoginRequiredMixin, CreateView):
    model = ReportType
    form_class = ReportTypeForm


class ReportTypeUpdate(LoginRequiredMixin, UpdateView):
    model = ReportType
    form_class = ReportTypeForm


class ReportTypeDelete(LoginRequiredMixin, DeleteView):
    model = ReportType
    success_url = reverse_lazy('report_types')


# 2. Проект
class SourceListView(LoginRequiredMixin, generic.ListView):
    model = Source


class SourceDetailView(LoginRequiredMixin, generic.DetailView):
    model = Source


class SourceCreate(LoginRequiredMixin, CreateView):
    model = Source
    form_class = SourceForm


class SourceUpdate(LoginRequiredMixin, UpdateView):
    model = Source
    form_class = SourceForm


class SourceDelete(LoginRequiredMixin, DeleteView):
    model = Source
    success_url = reverse_lazy('sources')


# 3. Местрождение
class RegionListView(LoginRequiredMixin, generic.ListView):
    model = Region


class RegionDetailView(LoginRequiredMixin, generic.DetailView):
    model = Region


class RegionCreate(LoginRequiredMixin, CreateView):
    model = Region
    form_class = RegionForm


class RegionUpdate(LoginRequiredMixin, UpdateView):
    model = Region
    form_class = RegionForm


class RegionDelete(LoginRequiredMixin, DeleteView):
    model = Region
    success_url = reverse_lazy('regions')


# 4. Отчеты
class ReportListView(LoginRequiredMixin, generic.ListView):
    model = Report

# class ReportListView(LoginRequiredMixin, FilterView):
#     model = Report
#     template_name = 'database/report_list.html'
#     paginate_by = 20

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['filter'] = ReportsFilter(self.request.GET, queryset=self.get_queryset())
#         if self.request.GET:
#             querystring = self.request.GET.copy()
#             if self.request.GET.get('page'):
#                 del querystring['page']
#             context['querystring'] = querystring.urlencode()
#         return context


class ReportDetailView(LoginRequiredMixin, generic.DetailView):
    model = Report

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        report_id = context['report'].__dict__['id']
        # Add in a QuerySet of all the books
        context['chapter_list'] = Chapter.objects.filter(report = report_id).order_by('id')
        return context

class ReportCreate(LoginRequiredMixin, CreateView):
    model = Report
    form_class = ReportForm


class ReportUpdate(LoginRequiredMixin, UpdateView):
    model = Report
    form_class = ReportForm


class ReportDelete(LoginRequiredMixin, DeleteView):
    model = Report
    success_url = reverse_lazy('reports')


# 4. Отчеты
class ChapterListView(LoginRequiredMixin, generic.ListView):
    model = Chapter


class ChapterDetailView(LoginRequiredMixin, generic.DetailView):
    model = Chapter


class ChapterCreate(LoginRequiredMixin, CreateView):
    model = Chapter
    form_class = ChapterForm


class ChapterUpdate(LoginRequiredMixin, UpdateView):
    model = Chapter
    form_class = ChapterForm


class ChapterDelete(LoginRequiredMixin, DeleteView):
    model = Chapter
    success_url = reverse_lazy('chapters')


class SearchView(LoginRequiredMixin, ListView):
    template_name = 'search.html'
    paginate_by = 20
    count = 0

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)
        context['count'] = self.count or 0
        context['query'] = self.request.GET.get('q')
        return context

    def get_queryset(self):
        request = self.request
        query = request.GET.get('q', None)

        if query is not None:
            specimen_results = Specimen.objects.search(query)
            property_results = Property.objects.search(query)

            # combine querysets
            queryset_chain = chain(
                specimen_results,
                property_results
            )
            qs = sorted(queryset_chain,
                        key=lambda instance: instance.pk,
                        reverse=True)
            self.count = len(qs)  # since qs is actually a list
            return qs
        return Property.objects.none()  # just an empty queryset as default


def generate_docx_file(object_id):

    report = Report.objects.get(pk=object_id)

    chapter_list = Chapter.objects.filter(report = report).order_by('id')

    all_chapters = f'\n'
    all_chapters += f'\n# {report.report_name}'

    br = '''\\
    \\
    \\'''

    plot_counter = 0
    for chapter in chapter_list:
        all_chapters += f'\n\n{chapter.chapter_text}\n\n'
        if chapter.graphics:
            plot_counter += 1
            object_id = chapter.id
            all_chapters += f'\n\n![Рисунок {plot_counter} - {chapter.plot_name}](./documents/graphics/buffer{object_id}.png)\n\n'


    docx_text = pypandoc.convert_text(all_chapters, 'docx', format='md', outputfile="temp.docx")
    document = Document('temp.docx')
    
    return document


def export_to_docx(request, object_id):

    report = Report.objects.get(pk=object_id)

    document = generate_docx_file(object_id)
    # Save the document to an in-memory stream
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)

    # Return the document as an HTTP response
    response = HttpResponse(buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    response['Content-Disposition'] = f'attachment; filename="report{report.id}.docx"'
    return response

def export_to_pdf(request, object_id):


    report = Report.objects.get(pk=object_id)

    document = generate_docx_file(object_id)
    # Save the document to an in-memory stream
    temp_doc_path = 'documents/temp_document.docx'
    temp_pdf_path = 'documents/temp_document.pdf'

    document.save(temp_doc_path)
    
    # Convert the docx file to PDF
    output_pdf = pypandoc.convert_file(
        source_file=temp_doc_path,
        format='docx',
        to='pdf',
        outputfile=temp_pdf_path,
        extra_args=['--pdf-engine=xelatex', '-V','mainfont="Arial"','-V','geometry:margin=1in']
    )

    with open(temp_pdf_path, 'rb') as f:
        pdf_content = f.read()

    # Return the PDF as an HTTP response
    response = HttpResponse(pdf_content, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report{report.id}.pdf"'
    return response


def faq(request):
    return render(request, 'faq.html')