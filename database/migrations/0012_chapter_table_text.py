# Generated by Django 4.1.4 on 2024-06-23 12:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0011_alter_chapter_chapter_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='chapter',
            name='table_text',
            field=models.TextField(blank=True, null=True),
        ),
    ]