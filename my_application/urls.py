"""
URL configuration for my_application project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('database/', include('database.urls')),
    path('', RedirectView.as_view(url='/database/', permanent=True)),
    path('accounts/', include('django.contrib.auth.urls')),
    # Add URL maps to redirect the base URL to the main application
    path('', RedirectView.as_view(url='/database/', permanent=True)),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
# Use static() to add url mapping to serve static files during development (only)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

