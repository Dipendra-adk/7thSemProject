from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.urls import reverse
from django.db import IntegrityError
from django.views.decorators.http import require_POST
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Property, ContactMessage, PropertyImage, Message,User
from django.core.exceptions import ValidationError
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
import re
import os
import numpy as np
import pandas as pd
import pickle
from django.http import HttpResponse
import json
from joblib import load
from pathlib import Path
from django.core.mail import EmailMessage
from house.ml_models.svm_model import SVR
from house.ml_models.decision_tree import DecisionTreeRegressor

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


BASE_DIR = Path(__file__).resolve().parent
print("Current directory:", os.getcwd())
print("File location:", BASE_DIR)
print("Model directory:", os.path.join(BASE_DIR, 'ml_models', 'saved_models'))
def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if not re.search(r'[a-zA-Z]', password) or not re.search(r'\d', password):
        return False, "Password must contain both letters and numbers."
    
    return True, ""

def is_admin(user):
    return user.is_staff or user.is_superuser

def signup_view(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        is_valid, error_message = validate_password(password)
        if not is_valid:
            messages.error(request, error_message)
            return redirect('signup')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already in use.")
            return redirect('signup')

        user = User.objects.create_user(
            email=email,
            password=password,
            first_name=name,
            phone=phone,
            is_active=False
        )

        login(request, user)  

        current_site = get_current_site(request)
        mail_subject = 'Activate your account'
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        activation_link = f"http://{current_site.domain}{reverse('activate', kwargs={'uidb64': uid, 'token': token})}"
        message = f"Hi {user.first_name},\n\nPlease activate your account by clicking the link below:\n\n{activation_link}"
        send_mail(mail_subject, message, 'noreply@cityestate.com', [email])
        messages.success(request, 'Account created successfully! Please confirm your email to complete the registration.')
        return redirect('login')  
    
    return render(request, 'signup.html')


def activate_view(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        messages.success(request, 'Thank you for confirming your email. You can now log in.')
        return redirect('login')
    else:
        messages.error(request, 'The activation link is invalid or expired.')
        return redirect('signup')


def login_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, 'Your account is not activated. Please check your email.')
                return redirect('login')
        else:
            messages.error(request, 'Invalid email or password')
            return redirect('login')

    return render(request, 'login.html')


# Logout view
@login_required
def logout_view(request):
    logout(request)
    next_url = request.GET.get('next', 'login')  
    return redirect(next_url)

# Password Reset View
def password_reset_view(request):
    if request.method == 'POST':
        email = request.POST['email']
        user = User.objects.filter(email=email).first()
        
        if user:
            current_site = get_current_site(request)
            mail_subject = 'Reset your password'
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            reset_link = f"http://{current_site.domain}{reverse('password_reset_confirm_view', kwargs={'uidb64': uid, 'token': token})}"
            message = f"Hi {user.username},\n\nClick the link below to reset your password:\n{reset_link}"
            send_mail(mail_subject, message, 'noreply@example.com', [email])
            
            messages.success(request, "A password reset link has been sent to your email.")
        else:
            messages.info(request, "If an account with this email exists, a reset link has been sent.")
        
        return render(request, 'password_reset.html')
    return render(request, 'password_reset.html')

def password_reset_confirm_view(request, uidb64, token):
    print(f"Received uidb64: {uidb64}, token: {token}") 
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        print(f"Decoded UID: {uid}")  # Debug
        user = User.objects.get(pk=uid)
        print(f"User: {user}")  # Debug
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        print(f"Error: {e}") # Debug
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            new_password = request.POST['new_password']
            user.set_password(new_password)
            user.save()
            messages.success(request, "Your password has been successfully reset. You can now log in.")
            return redirect('login')
        return render(request, 'password_reset_confirm.html', {'valid_link': True, 'uidb64': uidb64, 'token': token })
    else:
        messages.error(request, "The password reset link is invalid or has expired.")
        return render(request, 'password_reset_confirm.html', {'valid_link': False})


def home(request):
    query_title = request.GET.get('title', '')
    query_city = request.GET.get('city', '')
    
    properties = Property.objects.filter(is_approved=True)

    if query_title:
        properties = properties.filter(title__icontains=query_title)
    
    if query_city:
        properties = properties.filter(city__icontains=query_city)
    
    properties = properties.order_by('-created_at')[:4]
    
    return render(request, 'home.html', {
        'properties': properties,
        'query_title': query_title,
        'query_city': query_city,
    })

def user_dashboard(request):
    return render(request, "user_dashboard.html")

def admin_dashboard(request):
    pending_properties = Property.objects.filter(is_approved=False)  # Fetch unapproved properties
    return render(request, "admin_dashboard.html", {"properties": pending_properties})


def buyer(request):
    properties = Property.objects.filter(is_approved=True).order_by('-created_at')
    return render(request, 'buyer.html', {
        'properties': properties
    })
@user_passes_test(lambda u: u.is_superuser)
def approve_property(request, property_id):
    """Superuser approves a property"""
    property_obj = Property.objects.get(id=property_id)
    property_obj.is_approved = True
    property_obj.save()
    return redirect("buyer")  # Redirect to pending properties page


def decline_property(request, property_id):
    property = get_object_or_404(Property, id=property_id)
    property.delete()  # This removes the property from the database
    return redirect('admin_dashboard')  # Redirect back to the admin dashboard

@user_passes_test(lambda u: u.is_superuser)
def pending_properties(request):
    """List properties pending approval"""
    properties = Property.objects.filter(is_approved=False)  # Unapproved properties
    return render(request, "dashboard.html", {"properties": properties})

@login_required
def seller_view(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        city = request.POST.get('city')
        area = request.POST.get('area')
        bedrooms = request.POST.get('bedrooms')
        bathrooms = request.POST.get('bathrooms')
        stories = request.POST.get('stories')
        mainroad = request.POST.get('mainroad') == 'yes'
        guestroom = request.POST.get('guestroom') == 'yes'
        basement = request.POST.get('basement') == 'yes'
        hotwaterheating = request.POST.get('hotwaterheating') == 'yes'
        airconditioning = request.POST.get('airconditioning') == 'yes'
        parking = request.POST.get('parking') == 'yes'
        furnishingstatus = request.POST.get('furnishingstatus')
        price = request.POST.get('price')
        property_images = request.FILES.getlist('property_images')
        try:
            if not title or not city or not area or not bedrooms or not bathrooms or not stories or not price:
                raise ValidationError('Please fill out all required fields.')

            property = Property(
                title=title,
                city=city,
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                stories=stories,
                mainroad=mainroad,
                guestroom=guestroom,
                basement=basement,
                hotwaterheating=hotwaterheating,
                airconditioning=airconditioning,
                parking=parking,
                furnishingstatus=furnishingstatus,
                price=price,
                seller=request.user  
            )
            property.save()

            for image in property_images:
                property_image = PropertyImage(property=property, image=image)
                property_image.save()
            messages.success(request, 'Property listed and request sent to admin for approval.')
            return redirect('seller_view')
        except ValidationError as e:
            messages.error(request, e.message)
        except Exception as e:
            messages.error(request, 'Error saving property: ' + str(e))

    return render(request, 'seller.html')



def property_detail(request, property_id):
    property_obj = get_object_or_404(Property, id=property_id)
    seller_email = property_obj.seller.email
    
    if request.method == "POST":
        sender_name = request.POST.get("sender_name")
        sender_email = request.POST.get("sender_email")
        content = request.POST.get("content")
        
        if sender_name and sender_email and content:
            message = Message(
                sender_name=sender_name,
                sender_email=sender_email,
                content=content,
                property=property_obj,
            )
            message.save()
            
            subject = f"Inquiry about {property_obj.title}"
            body = f"""
            Hello {property_obj.seller.username},\n\n
            You have a new inquiry about your property: {property_obj.title}
            Now you can directly contact the buyer at {sender_email}.
            Contact him/her to discuss further details and discussion.\n
            Buyer Details:
            Name: {sender_name}
            Email: {sender_email}
            Message: {content}\n
            Please respond to the buyer if you're interested.
            """
            
            email = EmailMessage(
                subject=subject,
                body=body,
                from_email="noreply@citystate.com", 
                to=[seller_email],
                reply_to=[sender_email]
            )
            email.send(fail_silently=False)
            
            messages.success(request, "Your message has been sent to the seller.")
            return redirect("property_detail", property_id=property_id)
        else:
            error_message = "Please fill in all fields."
            return render(
                request,
                "property_detail.html",
                {"property": property_obj, "error_message": error_message},
            )
            
    return render(request, "property_detail.html", {"property": property_obj, "error_message": None})


def predict(request):
    try:
        # Load models and scalers
        feature_scaler = pickle.load(open('house/ml_models/saved_models/feature_scaler.pkl', 'rb'))
        svm_model = pickle.load(open('house/ml_models/saved_models/svm_model.pkl', 'rb'))
        dt_model = pickle.load(open('house/ml_models/saved_models/decision_tree.pkl', 'rb'))
        
        with open('house/ml_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        if request.method == 'POST':
            # Get input values with validation
            try:
                area = float(request.POST.get('area', 0))
                stories = float(request.POST.get('stories', 0))
                road_width = float(request.POST.get('road_width', 0))
                
                # Input validation
                if area <= 0 or stories <= 0 or road_width <= 0:
                    raise ValueError("Input values must be positive numbers")
                
            except ValueError as e:
                return render(request, 'predict.html', {'error': str(e)})
            
            city = request.POST.get('city')
            road_type = request.POST.get('road_type')
            
            # Create input data dictionary
            input_data = {
                'Floors': stories,
                'Area': area,
                'Road_Width': road_width,
                'City_Bhaktapur': 1 if city == 'Bhaktapur' else 0,
                'City_Kathmandu': 1 if city == 'Kathmandu' else 0,
                'City_Lalitpur': 1 if city == 'Lalitpur' else 0,
                'Road_Type_Blacktopped': 1 if road_type == 'Blacktopped' else 0,
                'Road_Type_Gravelled': 1 if road_type == 'Gravelled' else 0,
                'Road_Type_Soil Stabilized': 1 if road_type == 'Soil_Stabilized' else 0,
            }
            
            # Create DataFrame
            df = pd.DataFrame([input_data])
            df = df.reindex(columns=feature_names, fill_value=0)
            
            # Scale features
            X_scaled = feature_scaler.transform(df)
            
            # Get predictions (log scale)
            svm_pred_log = svm_model.predict(X_scaled)[0]
            dt_pred_log = dt_model.predict(X_scaled)[0]
            
            # Transform to actual prices
            svm_pred_price = np.expm1(svm_pred_log)
            dt_pred_price = np.expm1(dt_pred_log)
            
            # Apply reasonable price thresholds for Nepal real estate market
            # MIN_PRICE = 20000000  # 1 million NPR
            # MAX_PRICE = 500000000  # 500 million NPR
            
            svm_pred_price = np.clip(svm_pred_price)
            dt_pred_price = np.clip(dt_pred_price)
            
           
            
            return render(request, 'predict.html', {
                'prediction_svm': f'Rs. {svm_pred_price:,.2f}',
                'prediction_dt': f'Rs. {dt_pred_price:,.2f}',
                'form_data': request.POST,
                'input_area': area,
                'input_stories': stories,
                'input_road_width': road_width,
                'input_city': city,
                'input_road_type': road_type
            })
        
        return render(request, 'predict.html')
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render(request, 'predict.html', {
            'error': f"An error occurred: {str(e)}"
        })
    
def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        if name and email and subject and message:
            ContactMessage.objects.create(
                name=name,
                email=email,
                subject=subject,
                message=message
            )
            messages.success(request, "Your message has been sent successfully!")
            return redirect('contact') 
        else:
            messages.error(request, "Please fill in all fields.")

    return render(request, 'contact.html')
