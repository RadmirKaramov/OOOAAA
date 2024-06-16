import django_filters
from .models import Report



class ReportsFilter(django_filters.FilterSet):

    CHOICES = (
        ('ascending', 'По убыванию'),
        ('descending', 'По возрастанию'),
    )

    # ordering = django_filters.ChoiceFilter(label='Ordering', choices=CHOICES, method='filter_by_order')

    class Meta:
        model = Report
        exclude = ['plot_files', 'files']

    # def filter_by_order(self, queryset, name, value):
    #     expression = 'created' if value == 'ascending' else '-created'
    #     return queryset.order_by(expression)