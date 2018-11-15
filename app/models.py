from django.db import models


# Create your models here.
class PageHeader(models.Model):
    image = models.ImageField(upload_to='img')
