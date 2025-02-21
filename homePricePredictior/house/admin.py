from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, ContactMessage, Property, PropertyImage, Message, User, HousePricePrediction, MLModel
from django.utils.html import format_html

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
    list_display = ('title', 'city', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking','furnishingstatus', 'price', 'created_at')
    list_filter = ('city', 'bedrooms', 'bathrooms', 'furnishingstatus')
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


@admin.register(HousePricePrediction)
class HousePricePredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'area', 'stories', 'city', 'formatted_svm_prediction', 
                   'formatted_dt_prediction', 'created_at')
    list_filter = ('city', 'road_type', 'created_at')
    search_fields = ('city', 'area')
    readonly_fields = ('created_at', 'svm_prediction', 'dt_prediction')
    
    fieldsets = (
        ('Property Details', {
            'fields': (
                'area',
                'stories',
                'road_width',
                'city',
                'road_type'
            )
        }),
        ('Model Predictions', {
            'fields': (
                'svm_prediction',
                'dt_prediction'
            ),
            'classes': ('wide',)
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def formatted_svm_prediction(self, obj):
        if obj.svm_prediction is not None:
            return format_html('Rs. {:,.2f}', obj.svm_prediction)
        return '-'
    formatted_svm_prediction.short_description = 'SVM Prediction'
    
    def formatted_dt_prediction(self, obj):
        if obj.dt_prediction is not None:
            return format_html('Rs. {:,.2f}', obj.dt_prediction)
        return '-'
    formatted_dt_prediction.short_description = 'DT Prediction'

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'is_active', 'uploaded_at')
    list_filter = ('model_type', 'is_active', 'uploaded_at')
    search_fields = ('name',)
    readonly_fields = ('uploaded_at',)
    
    fieldsets = (
        (None, {
            'fields': ('name', 'model_type', 'model_file', 'is_active')
        }),
        ('Metadata', {
            'fields': ('uploaded_at',),
            'classes': ('collapse',)
        }),
    )
    
    def save_model(self, request, obj, form, change):
        # If this model is being set as active, deactivate other models of same type
        if obj.is_active:
            MLModel.objects.filter(
                model_type=obj.model_type
            ).exclude(id=obj.id).update(is_active=False)
        super().save_model(request, obj, form, change)