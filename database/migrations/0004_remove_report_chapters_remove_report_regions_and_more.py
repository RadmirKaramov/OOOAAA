# Generated by Django 4.1.4 on 2024-06-15 16:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0003_remove_report_chapter_name_1_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='report',
            name='chapters',
        ),
        migrations.RemoveField(
            model_name='report',
            name='regions',
        ),
        migrations.RemoveField(
            model_name='report',
            name='sources',
        ),
        migrations.AddField(
            model_name='reporttype',
            name='chapters',
            field=models.CharField(default='default', max_length=10000),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='reporttype',
            name='prompt',
            field=models.CharField(max_length=10000, null=True),
        ),
        migrations.AddField(
            model_name='reporttype',
            name='regions',
            field=models.ManyToManyField(to='database.region'),
        ),
        migrations.AddField(
            model_name='reporttype',
            name='report_type_description',
            field=models.CharField(default='default', max_length=50),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='reporttype',
            name='sources',
            field=models.ManyToManyField(to='database.source'),
        ),
        migrations.AddField(
            model_name='source',
            name='source_feature',
            field=models.CharField(default='default', max_length=40),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='reporttype',
            name='report_type',
            field=models.CharField(max_length=50),
        ),
    ]
