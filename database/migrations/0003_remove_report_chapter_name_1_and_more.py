# Generated by Django 4.1.4 on 2024-06-15 16:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0002_alter_report_chapter_text_1_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='report',
            name='chapter_name_1',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_name_2',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_name_3',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_name_4',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_name_5',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_name_6',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_text_1',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_text_2',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_text_3',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_text_4',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_text_5',
        ),
        migrations.RemoveField(
            model_name='report',
            name='chapter_text_6',
        ),
        migrations.AlterField(
            model_name='report',
            name='regions',
            field=models.ManyToManyField(to='database.region'),
        ),
        migrations.AlterField(
            model_name='report',
            name='sources',
            field=models.ManyToManyField(to='database.source'),
        ),
        migrations.CreateModel(
            name='Chapter',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('chapter_name', models.CharField(max_length=200)),
                ('chapter_text', models.CharField(max_length=20000)),
                ('chapter_validation', models.CharField(max_length=20000)),
                ('report', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='database.report')),
            ],
        ),
    ]