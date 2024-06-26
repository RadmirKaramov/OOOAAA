# Generated by Django 4.2.13 on 2024-06-06 19:43

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Region',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('region', models.CharField(max_length=20)),
                ('region_feature', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='ReportType',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('report_type', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='Source',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source', models.CharField(max_length=40)),
                ('URL', models.CharField(max_length=40)),
            ],
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('report_name', models.CharField(max_length=200, null=True)),
                ('short_description', models.CharField(max_length=200, null=True)),
                ('additional_sources', models.CharField(max_length=200, null=True)),
                ('filters', models.CharField(max_length=200, null=True)),
                ('plot_files', models.FileField(blank=True, null=True, upload_to='documents/graphics')),
                ('files', models.FileField(blank=True, null=True, upload_to='documents/files')),
                ('chapters', models.CharField(max_length=200, null=True)),
                ('graphics', models.FileField(blank=True, null=True, upload_to='documents/graphics')),
                ('chapter_name_1', models.CharField(max_length=200, null=True)),
                ('chapter_text_1', models.CharField(max_length=2000, null=True)),
                ('chapter_name_2', models.CharField(max_length=200, null=True)),
                ('chapter_text_2', models.CharField(max_length=2000, null=True)),
                ('chapter_name_3', models.CharField(max_length=200, null=True)),
                ('chapter_text_3', models.CharField(max_length=2000, null=True)),
                ('chapter_name_4', models.CharField(max_length=200, null=True)),
                ('chapter_text_4', models.CharField(max_length=2000, null=True)),
                ('chapter_name_5', models.CharField(max_length=200, null=True)),
                ('chapter_text_5', models.CharField(max_length=2000, null=True)),
                ('chapter_name_6', models.CharField(max_length=200, null=True)),
                ('chapter_text_6', models.CharField(max_length=2000, null=True)),
                ('regions', models.ManyToManyField(null=True, to='database.region')),
                ('report_type', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='database.reporttype')),
                ('sources', models.ManyToManyField(null=True, to='database.source')),
            ],
        ),
    ]
