import re
import xlrd

import numpy as np

# from .models import Project, Organization, Result, Property, Specimen


def handle_uploaded_properties(request, f):
    book = xlrd.open_workbook(file_contents=f.read())
    max_row = book.sheets()[0].nrows

    for row in range(max_row):
        _, created = Property.objects.get_or_create(
            property=book.sheets()[0].cell(row, 0).value,
            constant_property=book.sheets()[0].cell(row, 1).value,
            )


def handle_uploaded_tests(request, f):
    book = xlrd.open_workbook(file_contents=f.read())

    max_row = book.sheets()[0].nrows - 9

    organization_v = book.sheets()[0].cell(2, 1).value
    organization_get = Organization.objects.get(organization=str(organization_v))

    project_v = book.sheets()[0].cell(3, 1).value
    project_get = Project.objects.get(project=str(project_v))

    program_v = book.sheets()[0].cell(4, 1).value
    program_get = TestProgram.objects.get(program_number=int(program_v))

    protocol_v = book.sheets()[0].cell(5, 1).value
    protocol_get = Protocol.objects.get(protocol_number=str(protocol_v))

    properties_get = list()

    for n in range(Property.objects.all().count()):
        property_v = book.sheets()[0].cell(7, 4 + n).value
        properties_get.append(Property.objects.get(property=property_v))


    for i in range(max_row):

        # result_number_v = book.sheets()[0].cell(i + 9, 0).value
        passport_v = int(book.sheets()[0].cell(i + 9, 1).value)
        if SpecimenPass.objects.filter(material_pass=protocol_get.material.__dict__['id']).filter(pass_number=int(passport_v)):
            specimenpass_get = SpecimenPass.objects.filter(material_pass=protocol_get.material.__dict__['id']).get(pass_number=int(passport_v))
        else:
            # следущее условие необходимо для создание маски в нумерации паспортов образцов
            if int(passport_v) < 10:
                pass_number_var = '0' + str(int(passport_v)),
            else:
                pass_number_var = str(int(passport_v)),
                _, created_spec = SpecimenPass.objects.get_or_create(
                    pass_number = pass_number_var[0],
                    amount = 10,
                    project=project_get,
                    program=program_get,
                    material_pass=protocol_get.material,

                    pass_file = 'documents/specimenpass/паспортОбразцов.txt'
                )
                specimenpass_get = SpecimenPass.objects.filter(material_pass=protocol_get.material.__dict__['id']).get(pass_number=pass_number_var[0])


        specimen_number_v = book.sheets()[0].cell(i + 9, 2).value
        status_v = book.sheets()[0].cell(i + 9, 3).value
        if Specimen.objects.filter(specimen_pass=specimenpass_get).filter(specimen_number=str(specimen_number_v)):
            specimen_get = Specimen.objects.filter(specimen_pass=specimenpass_get).get(specimen_number=int(specimen_number_v))
        else:
            status_get = SpecimenStatus.objects.get(status=status_v)
            _, created_spec = Specimen.objects.get_or_create(
                specimen_pass=specimenpass_get,
                specimen_number=int(specimen_number_v),
                status=status_get
            )
            specimen_get = Specimen.objects.filter(specimen_pass=specimenpass_get).get(specimen_number=int(specimen_number_v))
        
        value_v = list()
        chrono_v = book.sheets()[0].cell(i + 9, 4 + Property.objects.all().count()).value

        for n in range(Property.objects.all().count()):
        
            value_v = book.sheets()[0].cell(i + 9, 4 + n).value

            if value_v != '':
                if properties_get[n].__dict__['constant_property'] is True:
                    last_result = ConstantProperty.objects.filter(specimen_number=specimen_get).order_by('id').last()

                    if last_result is not None:
                        next_number = str(int(last_result.constant_property_number) + 1)
                    else:
                        next_number = int(1)

                    _, created = ConstantProperty.objects.get_or_create(
                        constant_property_number=int(next_number),
                        constant_property=properties_get[n],
                        specimen_number=specimen_get,
                        property_value=value_v,

                        )
                else:
                    last_result = Result.objects.filter(specimen_number=specimen_get).order_by('id').last()

                    if last_result is not None:
                        next_number = str(int(last_result.result_number) + 1)
                    else:
                        next_number = int(1)

                    _, created = Result.objects.get_or_create(
                        result_number=int(next_number),
                        property=properties_get[n],
                        specimen_number=specimen_get,
                        result=value_v,
                        chronometer=chrono_v,
                        )