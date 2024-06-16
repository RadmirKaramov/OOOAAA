from django.shortcuts import render
from django.views import generic
from .models import *
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from .uploads import handle_uploaded_tests
from .forms import *
from itertools import chain
from django.views.generic import ListView
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required


@login_required(login_url='/accounts/login/')
def index(request):
    """View function for home page of site."""

    # Generate counts of some of the main objects
    num_visits = request.session.get('num_visits', 0)
    num_projects = Project.objects.all().count()
    num_organizations = Organization.objects.all().count()
    num_materials = Material.objects.all().count()
    num_specimen = Specimen.objects.all().count()
    num_result = Result.objects.all().count()
    request.session['num_visits'] = num_visits + 1

    context = {
        'num_projects': num_projects,
        'num_organizations': num_organizations,
        'num_materials': num_materials,
        'num_test': num_result,
        'num_specimen': num_specimen,
        'num_visits': num_visits,
    }

    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)



# Organization
def organization_form_upload(request):
    if request.method == 'POST':
        form = OrganizationForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('organization-detail')
    else:
        form = OrganizationForm()
    return render(request, 'database/well_form_upload.html', {
        'form': form
    })


@login_required(login_url='/accounts/login/')
class OrganizationListView(generic.ListView):
    model = Organization


@login_required(login_url='/accounts/login/')
class OrganizationDetailView(generic.DetailView):
    model = Organization


@login_required(login_url='/accounts/login/')
class OrganizationCreate(CreateView):
    model = Organization
    form_class = OrganizationForm


@login_required(login_url='/accounts/login/')
class OrganizationUpdate(UpdateView):
    model = Organization
    form_class = OrganizationForm


@login_required(login_url='/accounts/login/')
class OrganizationDelete(DeleteView):
    model = Organization
    success_url = reverse_lazy('organizations')


# TestingMachine
@login_required(login_url='/accounts/login/')
class TestingMachineListView(generic.ListView):
    model = TestingMachine


@login_required(login_url='/accounts/login/')
class TestingMachineDetailView(generic.DetailView):
    model = TestingMachine


@login_required(login_url='/accounts/login/')
class TestingMachineCreate(CreateView):
    model = TestingMachine
    form_class = TestingMachineForm


@login_required(login_url='/accounts/login/')
class TestingMachineUpdate(UpdateView):
    model = TestingMachine
    form_class = TestingMachineForm


@login_required(login_url='/accounts/login/')
class TestingMachineDelete(DeleteView):
    model = TestingMachine
    success_url = reverse_lazy('testingmachines')


# MetrologyBase
@login_required(login_url='/accounts/login/')
class MetrologyBaseListView(generic.ListView):
    model = MetrologyBase


@login_required(login_url='/accounts/login/')
class MetrologyBaseDetailView(generic.DetailView):
    model = MetrologyBase


@login_required(login_url='/accounts/login/')
class MetrologyBaseCreate(CreateView):
    model = MetrologyBase
    form_class = MetrologyBaseForm


@login_required(login_url='/accounts/login/')
class MetrologyBaseUpdate(UpdateView):
    model = MetrologyBase
    form_class = MetrologyBaseForm


@login_required(login_url='/accounts/login/')
class MetrologyBaseDelete(DeleteView):
    model = MetrologyBase
    success_url = reverse_lazy('metrologybases')


# Computer
@login_required(login_url='/accounts/login/')
class ComputerListView(generic.ListView):
    model = Computer


@login_required(login_url='/accounts/login/')
class ComputerDetailView(generic.DetailView):
    model = Computer


@login_required(login_url='/accounts/login/')
class ComputerCreate(CreateView):
    model = Computer
    form_class = ComputerForm


@login_required(login_url='/accounts/login/')
class ComputerUpdate(UpdateView):
    model = Computer
    form_class = ComputerForm


@login_required(login_url='/accounts/login/')
class ComputerDelete(DeleteView):
    model = Computer
    success_url = reverse_lazy('computers')


# Fixture
@login_required(login_url='/accounts/login/')
class FixtureListView(generic.ListView):
    model = Fixture


@login_required(login_url='/accounts/login/')
class FixtureDetailView(generic.DetailView):
    model = Fixture


@login_required(login_url='/accounts/login/')
class FixtureCreate(CreateView):
    model = Fixture
    form_class = FixtureForm


@login_required(login_url='/accounts/login/')
class FixtureUpdate(UpdateView):
    model = Fixture
    form_class = FixtureForm


@login_required(login_url='/accounts/login/')
class FixtureDelete(DeleteView):
    model = Fixture
    success_url = reverse_lazy('fixtures')


# Standard
@login_required(login_url='/accounts/login/')
class StandardListView(generic.ListView):
    model = Standard


@login_required(login_url='/accounts/login/')
class StandardDetailView(generic.DetailView):
    model = Standard


@login_required(login_url='/accounts/login/')
class StandardCreate(CreateView):
    model = Standard
    form_class = StandardForm


@login_required(login_url='/accounts/login/')
class StandardUpdate(UpdateView):
    model = Standard
    form_class = StandardForm


@login_required(login_url='/accounts/login/')
class StandardDelete(DeleteView):
    model = Standard
    success_url = reverse_lazy('standards')


# TestType
@login_required(login_url='/accounts/login/')
class TestTypeListView(generic.ListView):
    model = TestType


@login_required(login_url='/accounts/login/')
class TestTypeDetailView(generic.DetailView):
    model = TestType


@login_required(login_url='/accounts/login/')
class TestTypeCreate(CreateView):
    model = TestType
    form_class = TestTypeForm


@login_required(login_url='/accounts/login/')
class TestTypeUpdate(UpdateView):
    model = TestType
    form_class = TestTypeForm


@login_required(login_url='/accounts/login/')
class TestTypeDelete(DeleteView):
    model = TestType
    success_url = reverse_lazy('testtypes')


# Project
@login_required(login_url='/accounts/login/')
class ProjectListView(generic.ListView):
    model = Project


@login_required(login_url='/accounts/login/')
class ProjectDetailView(generic.DetailView):
    model = Project


@login_required(login_url='/accounts/login/')
class ProjectCreate(CreateView):
    model = Project
    form_class = ProjectForm


@login_required(login_url='/accounts/login/')
class ProjectUpdate(UpdateView):
    model = Project
    form_class = ProjectForm


@login_required(login_url='/accounts/login/')
class ProjectDelete(DeleteView):
    model = Project
    success_url = reverse_lazy('projects')


# Project
@login_required(login_url='/accounts/login/')
class TestProgramListView(generic.ListView):
    model = TestProgram


@login_required(login_url='/accounts/login/')
class TestProgramDetailView(generic.DetailView):
    model = TestProgram


@login_required(login_url='/accounts/login/')
class TestProgramCreate(CreateView):
    model = TestProgram
    form_class = TestProgramForm


@login_required(login_url='/accounts/login/')
class TestProgramUpdate(UpdateView):
    model = TestProgram
    form_class = TestProgramForm


@login_required(login_url='/accounts/login/')
class TestProgramDelete(DeleteView):
    model = TestProgram
    success_url = reverse_lazy('testprograms')


# Property
@login_required(login_url='/accounts/login/')
class PropertyListView(generic.ListView):
    model = Property


@login_required(login_url='/accounts/login/')
class PropertyDetailView(generic.DetailView):
    model = Property


@login_required(login_url='/accounts/login/')
class PropertyCreate(CreateView):
    model = Property
    form_class = PropertyForm


@login_required(login_url='/accounts/login/')
class PropertyUpdate(UpdateView):
    model = Property
    form_class = PropertyForm


@login_required(login_url='/accounts/login/')
class PropertyDelete(DeleteView):
    model = Property
    success_url = reverse_lazy('propertys')


# Material
@login_required(login_url='/accounts/login/')
class MaterialListView(generic.ListView):
    model = Material


@login_required(login_url='/accounts/login/')
class MaterialDetailView(generic.DetailView):
    model = Material


@login_required(login_url='/accounts/login/')
class MaterialCreate(CreateView):
    model = Material
    form_class = MaterialForm


@login_required(login_url='/accounts/login/')
class MaterialUpdate(UpdateView):
    model = Material
    form_class = MaterialForm


@login_required(login_url='/accounts/login/')
class MaterialDelete(DeleteView):
    model = Material
    success_url = reverse_lazy('materials')


# SpecimenPass
@login_required(login_url='/accounts/login/')
class SpecimenPassListView(generic.ListView):
    model = SpecimenPass


@login_required(login_url='/accounts/login/')
class SpecimenPassDetailView(generic.DetailView):
    model = SpecimenPass


@login_required(login_url='/accounts/login/')
class SpecimenPassCreate(CreateView):
    model = SpecimenPass
    form_class = SpecimenPassForm


@login_required(login_url='/accounts/login/')
class SpecimenPassUpdate(UpdateView):
    model = SpecimenPass
    form_class = SpecimenPassForm


@login_required(login_url='/accounts/login/')
class SpecimenPassDelete(DeleteView):
    model = SpecimenPass
    success_url = reverse_lazy('specimenpasss')


# Manufacturing
@login_required(login_url='/accounts/login/')
class ManufacturingListView(generic.ListView):
    model = Manufacturing


@login_required(login_url='/accounts/login/')
class ManufacturingDetailView(generic.DetailView):
    model = Manufacturing


@login_required(login_url='/accounts/login/')
class ManufacturingCreate(CreateView):
    model = Manufacturing
    form_class = ManufacturingForm


@login_required(login_url='/accounts/login/')
class ManufacturingUpdate(UpdateView):
    model = Manufacturing
    form_class = ManufacturingForm


@login_required(login_url='/accounts/login/')
class ManufacturingDelete(DeleteView):
    model = Manufacturing
    success_url = reverse_lazy('manufacturings')


# MetroMeasure
@login_required(login_url='/accounts/login/')
class MetroMeasureListView(generic.ListView):
    model = MetroMeasure


@login_required(login_url='/accounts/login/')
class MetroMeasureDetailView(generic.DetailView):
    model = MetroMeasure


@login_required(login_url='/accounts/login/')
class MetroMeasureCreate(CreateView):
    model = MetroMeasure
    form_class = MetroMeasureForm


@login_required(login_url='/accounts/login/')
class MetroMeasureUpdate(UpdateView):
    model = MetroMeasure
    form_class = MetroMeasureForm


@login_required(login_url='/accounts/login/')
class MetroMeasureDelete(DeleteView):
    model = MetroMeasure
    success_url = reverse_lazy('metromeasures')


# MetroMeasure
@login_required(login_url='/accounts/login/')
class PreparationTypeListView(generic.ListView):
    model = PreparationType


@login_required(login_url='/accounts/login/')
class PreparationTypeDetailView(generic.DetailView):
    model = PreparationType


@login_required(login_url='/accounts/login/')
class PreparationTypeCreate(CreateView):
    model = PreparationType
    form_class = PreparationTypeForm


@login_required(login_url='/accounts/login/')
class PreparationTypeUpdate(UpdateView):
    model = PreparationType
    form_class = PreparationTypeForm


@login_required(login_url='/accounts/login/')
class PreparationTypeDelete(DeleteView):
    model = PreparationType
    success_url = reverse_lazy('preparationtypes')


# SpecimenPreparation
@login_required(login_url='/accounts/login/')
class SpecimenPreparationListView(generic.ListView):
    model = SpecimenPreparation


@login_required(login_url='/accounts/login/')
class SpecimenPreparationDetailView(generic.DetailView):
    model = SpecimenPreparation


@login_required(login_url='/accounts/login/')
class SpecimenPreparationCreate(CreateView):
    model = SpecimenPreparation
    form_class = SpecimenPreparationForm


@login_required(login_url='/accounts/login/')
class SpecimenPreparationUpdate(UpdateView):
    model = SpecimenPreparation
    form_class = SpecimenPreparationForm


@login_required(login_url='/accounts/login/')
class SpecimenPreparationDelete(DeleteView):
    model = SpecimenPreparation
    success_url = reverse_lazy('preparationtypes')


# Protocol
@login_required(login_url='/accounts/login/')
class ProtocolListView(generic.ListView):
    model = Protocol


@login_required(login_url='/accounts/login/')
class ProtocolDetailView(generic.DetailView):
    model = Protocol


@login_required(login_url='/accounts/login/')
class ProtocolCreate(CreateView):
    model = Protocol
    form_class = ProtocolForm


@login_required(login_url='/accounts/login/')
class ProtocolUpdate(UpdateView):
    model = Protocol
    form_class = ProtocolForm


@login_required(login_url='/accounts/login/')
class ProtocolDelete(DeleteView):
    model = Protocol
    success_url = reverse_lazy('protocols')


# SpecimenControl
@login_required(login_url='/accounts/login/')
class SpecimenControlListView(generic.ListView):
    model = SpecimenControl


@login_required(login_url='/accounts/login/')
class SpecimenControlDetailView(generic.DetailView):
    model = SpecimenControl


@login_required(login_url='/accounts/login/')
class SpecimenControlCreate(CreateView):
    model = SpecimenControl
    form_class = SpecimenControlForm


@login_required(login_url='/accounts/login/')
class SpecimenControlUpdate(UpdateView):
    model = SpecimenControl
    form_class = SpecimenControlForm


@login_required(login_url='/accounts/login/')
class SpecimenControlDelete(DeleteView):
    model = SpecimenControl
    success_url = reverse_lazy('specimencontrols')


# Specimen
@login_required(login_url='/accounts/login/')
class SpecimenStatusListView(generic.ListView):
    model = SpecimenStatus


@login_required(login_url='/accounts/login/')
class SpecimenStatusDetailView(generic.DetailView):
    model = SpecimenStatus


@login_required(login_url='/accounts/login/')
class SpecimenStatusCreate(CreateView):
    model = SpecimenStatus
    form_class = SpecimenStatusForm


@login_required(login_url='/accounts/login/')
class SpecimenStatusUpdate(UpdateView):
    model = SpecimenStatus
    form_class = SpecimenStatusForm


@login_required(login_url='/accounts/login/')
class SpecimenStatusDelete(DeleteView):
    model = SpecimenStatus
    success_url = reverse_lazy('specimenstatuss')


# Specimen
@login_required(login_url='/accounts/login/')
class SpecimenListView(generic.ListView):
    model = Specimen


@login_required(login_url='/accounts/login/')
class SpecimenDetailView(generic.DetailView):
    model = Specimen


@login_required(login_url='/accounts/login/')
class SpecimenCreate(CreateView):
    model = Specimen
    form_class = SpecimenForm


@login_required(login_url='/accounts/login/')
class SpecimenUpdate(UpdateView):
    model = Specimen
    form_class = SpecimenForm


@login_required(login_url='/accounts/login/')
class SpecimenDelete(DeleteView):
    model = Specimen
    success_url = reverse_lazy('specimens')


# Deviation
@login_required(login_url='/accounts/login/')
class MachineDataListView(generic.ListView):
    model = MachineData


@login_required(login_url='/accounts/login/')
class MachineDataDetailView(generic.DetailView):
    model = MachineData


@login_required(login_url='/accounts/login/')
class MachineDataCreate(CreateView):
    model = MachineData
    form_class = MachineDataForm


@login_required(login_url='/accounts/login/')
class MachineDataUpdate(UpdateView):
    model = MachineData
    form_class = MachineDataForm


@login_required(login_url='/accounts/login/')
class MachineDataDelete(DeleteView):
    model = MachineData
    success_url = reverse_lazy('machinedatas')


# Deviation
@login_required(login_url='/accounts/login/')
class DeviationListView(generic.ListView):
    model = Deviation


@login_required(login_url='/accounts/login/')
class DeviationDetailView(generic.DetailView):
    model = Deviation


@login_required(login_url='/accounts/login/')
class DeviationCreate(CreateView):
    model = Deviation
    form_class = DeviationForm


@login_required(login_url='/accounts/login/')
class DeviationUpdate(UpdateView):
    model = Deviation
    form_class = DeviationForm


@login_required(login_url='/accounts/login/')
class DeviationDelete(DeleteView):
    model = Deviation
    success_url = reverse_lazy('deviations')


# Result
@login_required(login_url='/accounts/login/')
class ResultListView(generic.ListView):
    model = Result


@login_required(login_url='/accounts/login/')
class ResultDetailView(generic.DetailView):
    model = Result


@login_required(login_url='/accounts/login/')
class ResultCreate(CreateView):
    model = Result
    form_class = ResultForm


@login_required(login_url='/accounts/login/')
class ResultUpdate(UpdateView):
    model = Result
    form_class = ResultForm


@login_required(login_url='/accounts/login/')
class ResultDelete(DeleteView):
    model = Result
    success_url = reverse_lazy('results')

##############################################


class StudentBulkAddView(generic.ListView):
    model = Project
    template_name = 'project_add.html'

    def dispatch(self, *args, **kwargs):
        return super(StudentBulkAddView, self).dispatch(*args, **kwargs)

    def post(self, request):
        try:
            handle_uploaded_file(request, request.FILES['file'])
            success = True
        except:
            pass

        return render(request, 'project_add.html')


class UploadSpecimensView(generic.ListView):
    model = Specimen
    template_name = 'specimens_add.html'

    def dispatch(self, *args, **kwargs):
        return super(UploadSpecimensView, self).dispatch(*args, **kwargs)

    def post(self, request):
        try:
            handle_uploaded_tests(request, request.FILES['file'])
            success = True
        except:
            pass

        return render(request, 'specimens_add.html')


@login_required(login_url='/accounts/login/')
class SearchView(ListView):
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
            standard_results = Standard.objects.search(query)
            property_results = Property.objects.search(query)

            # combine querysets
            queryset_chain = chain(
                specimen_results,
                standard_results,
                property_results
            )
            qs = sorted(queryset_chain,
                        key=lambda instance: instance.pk,
                        reverse=True)
            self.count = len(qs)  # since qs is actually a list
            return qs
        return Property.objects.none()  # just an empty queryset as default
