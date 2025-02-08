from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.base_user import BaseUserManager


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        if password:
            user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        return self.create_user(email, password, **extra_fields)

class User(AbstractUser):
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    
    username = models.CharField(max_length=150, unique=True, null=True, blank=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    
    objects = CustomUserManager()
    
    groups = models.ManyToManyField(
        'auth.Group',
        related_name='house_user_set',
        blank=True
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='house_user_permissions',
        blank=True
    )

    def __str__(self):
        return self.email

    def save(self, *args, **kwargs):
        if not self.username:
            self.username = self.email
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'


class Property(models.Model):
    seller = models.ForeignKey(User, on_delete=models.CASCADE, related_name="properties_for_sale")
    CITY_CHOICES = [
        ('ktm', 'Kathmandu'),
        ('bkt', 'Bhaktapur'),
        ('lat', 'Lalitpur'),
    ]
    FURNISHING_STATUS_CHOICES = [
        ('furnished', 'Furnished'),
        ('semi_furnished', 'Semi-furnished'),
        ('unfurnished', 'Unfurnished'),
    ]
    title = models.CharField(max_length=200)
    city = models.CharField(max_length=100, choices=CITY_CHOICES)
    area = models.DecimalField(max_digits=10, decimal_places=2)
    bedrooms = models.IntegerField()
    bathrooms = models.IntegerField()
    stories = models.IntegerField()
    mainroad = models.BooleanField(default=False)
    guestroom = models.BooleanField(default=False)
    basement = models.BooleanField(default=False)
    hotwaterheating = models.BooleanField(default=False)
    airconditioning = models.BooleanField(default=False)
    parking = models.IntegerField(default=0)
    furnishingstatus = models.CharField(max_length=20, choices=FURNISHING_STATUS_CHOICES, default='unfurnished')
    price = models.DecimalField(max_digits=20, decimal_places=2)
    property_image = models.ImageField(upload_to='property_images/')
    is_approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        
        return f"Property in {self.city} - {self.area}"

class PropertyImage(models.Model):
    property = models.ForeignKey(Property, related_name="images", on_delete=models.CASCADE)
    image = models.ImageField(upload_to='property_images/')
    
    def __str__(self):
        return f"Image for {self.property.title}"

class Message(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE, null=True, blank=True)  
    sender_name = models.CharField(max_length=255)
    sender_email = models.EmailField()
    content = models.TextField()
    sent_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.sender_name} about {self.property.title if self.property else 'General Inquiry'}"

class ContactMessage(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField()
    subject = models.CharField(max_length=255)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.name} - {self.subject}"
