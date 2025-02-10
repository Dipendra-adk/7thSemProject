from django.urls import path
from house.views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('signup/', signup_view, name='signup'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('activate/<uidb64>/<token>/', activate_view, name='activate'),
    path('password-reset/', password_reset_view, name='password_reset'),
    path('password-reset-confirm/<uidb64>/<token>/', password_reset_confirm_view, name='password_reset_confirm_view'),
    path('', home, name='home'),
    path("buyer/", buyer, name='buyer'),
    path("seller/", seller_view, name='seller_view'),
    path('property/<int:property_id>/', property_detail, name='property_detail'),
    path("predict/",predict, name="predict"),
    path("contact/",contact, name='contact'), 
    path("buyer/pending/", pending_properties, name="pending_properties"),
    path("buyer/approve/<int:property_id>/", approve_property, name="approve_property"),
    path('buyer/decline/<int:property_id>/', decline_property, name='decline_property'),
    path("dashboard/", user_dashboard, name="user_dashboard"),
    path("admin-dashboard/", admin_dashboard, name="admin_dashboard"),
    path('view-dataset/<str:filename>/', view_dataset, name='view_dataset'),
    path('decline_property/<int:property_id>/', decline_property, name='decline_property'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)