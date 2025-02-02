from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, ContactMessage, Property, PropertyImage, Message, User

# Register the ContactMessage model
admin.site.register(ContactMessage)

# Register User model if you want to customize the admin for it
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'first_name', 'phone', 'location', 'is_staff', 'is_active')
    search_fields = ('email', 'first_name', 'phone')
    ordering = ('email',)
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'phone', 'location')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_staff', 'is_active')}
        ),
    )

# Register the Property model with custom admin
@admin.register(Property)
class PropertyAdmin(admin.ModelAdmin):
    list_display = ('title', 'city', 'area', 'bedrooms', 'bathrooms', 'floor_number', 'parking_space', 'year_built', 'building_area', 'road_width', 'road_type', 'price', 'created_at')
    list_filter = ('city', 'bedrooms', 'bathrooms', 'floor_number', 'road_type', 'parking_space')
    search_fields = ('title', 'city')

# Register PropertyImage model with custom admin
@admin.register(PropertyImage)
class PropertyImageAdmin(admin.ModelAdmin):
    list_display = ('property', 'image')
    search_fields = ('property__title', 'property__city')
    list_filter = ('property__city',)

# Register Message model with custom admin
@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('sender_name', 'sender_email', 'property', 'sent_at')
    list_filter = ('sent_at',)
    search_fields = ('sender_name', 'sender_email', 'content')
