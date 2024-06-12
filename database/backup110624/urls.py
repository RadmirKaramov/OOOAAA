from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

# from django.conf.urls import url //depricated
from django.contrib import admin

# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]

urlpatterns = [
    path('', views.index, name='index'),
]

urlpatterns += [
    path('report_types/', views.ReportTypeListView.as_view(), name='report_types'),
    path('report_types/<int:pk>', views.ReportTypeDetailView.as_view(), name='report_type-detail'),
    path('report_types/create/', views.ReportTypeCreate.as_view(), name='report_type_create'),
    path('report_types/<int:pk>/update/', views.ReportTypeUpdate.as_view(), name='report_type_update'),
    path('report_types/<int:pk>/delete/', views.ReportTypeDelete.as_view(), name='report_type_delete'),
]

urlpatterns += [
    path('sources/', views.SourceListView.as_view(), name='sources'),
    path('sources/<int:pk>', views.SourceDetailView.as_view(), name='source-detail'),
    path('sources/create/', views.SourceCreate.as_view(), name='source_create'),
    path('sources/<int:pk>/update/', views.SourceUpdate.as_view(), name='source_update'),
    path('sources/<int:pk>/delete/', views.SourceDelete.as_view(), name='source_delete'),
]

urlpatterns += [
    path('regions/', views.RegionListView.as_view(), name='regions'),
    path('regions/<int:pk>', views.RegionDetailView.as_view(), name='region-detail'),
    path('regions/create/', views.RegionCreate.as_view(), name='region_create'),
    path('regions/<int:pk>/update/', views.RegionUpdate.as_view(), name='region_update'),
    path('regions/<int:pk>/delete/', views.RegionDelete.as_view(), name='region_delete'),
]

urlpatterns += [
    path('reports/', views.ReportListView.as_view(), name='reports'),
    path('reports/<int:pk>', views.ReportDetailView.as_view(), name='report-detail'),
    path('reports/create/', views.ReportCreate.as_view(), name='report_create'),
    path('reports/<int:pk>/update/', views.ReportUpdate.as_view(), name='report_update'),
    path('reports/<int:pk>/delete/', views.ReportDelete.as_view(), name='report_delete'),
]


urlpatterns += [
    path('search', views.SearchView.as_view(), name='search'),
    path('export/docx/<int:object_id>/', views.export_to_docx, name='export_to_docx'),
    path('export/pdf/<int:object_id>/', views.export_to_pdf, name='export_to_pdf'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
