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
from django.core.files.storage import FileSystemStorage
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
from house.ml_models.svm_model import SVMRegressor
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
        send_mail(mail_subject, message, 'noreply@example.com', [email])
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
        print(f"Error: {e}")  # Debug
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
    
    properties = Property.objects.all()

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


def buyer(request):
    properties = Property.objects.all()
    return render(request, 'buyer.html', {
        'properties': properties
    })

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
        prefarea = request.POST.get('prefarea') == 'yes'
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
                prefarea=prefarea,
                furnishingstatus=furnishingstatus,
                price=price,
                seller=request.user
                
            )
            property.save()

            for image in property_images:
                property_image = PropertyImage(property=property, image=image)
                property_image.save()
            messages.success(request, 'Property listed successfully.')
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
            You have a new inquiry about your property: {property_obj.title}\n
            Now you can directly contact the buyer at {sender_email}.\n
            Contact him/her to discuss further details and discussion.\n\n
            Buyer Details:
            Name: {sender_name}
            Email: {sender_email}\n
            Message: {content}\n\n
            Please respond to the buyer if you're interested.
            """
            send_mail(
                subject,
                body,
                sender_email,  
                [seller_email],  
                fail_silently=False,
                
            )
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
        # Load necessary files
        with open('house/encoder_mappings.json', 'r') as f:
            encoder_mappings = json.load(f)
        
        with open('house/ml_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        scaler = load('house/ml_models/saved_models/scaler.pkl')
        svm_model = load('house/ml_models/saved_models/svm_model.pkl')
        dt_model = load('house/ml_models/saved_models/decision_tree.pkl')

        if request.method == 'POST':
            input_data = {
                'area': float(request.POST.get('area')),
                'bedrooms': int(request.POST.get('bedrooms')),
                'bathrooms': int(request.POST.get('bathrooms')),
                'stories': int(request.POST.get('stories')),
                'mainroad': encoder_mappings['mainroad']['encode'].get(request.POST.get('mainroad'), 0),
                'guestroom': encoder_mappings['guestroom']['encode'].get(request.POST.get('guestroom'), 0),
                'basement': encoder_mappings['basement']['encode'].get(request.POST.get('basement'), 0),
                'hotwaterheating': encoder_mappings['hotwaterheating']['encode'].get(request.POST.get('hotwaterheating'), 0),
                'airconditioning': encoder_mappings['airconditioning']['encode'].get(request.POST.get('airconditioning'), 0),
                'parking': int(request.POST.get('parking')),
                'prefarea': encoder_mappings['prefarea']['encode'].get(request.POST.get('prefarea'), 0),
                'furnishingstatus_furnished': 1 if request.POST.get('furnishingstatus') == 'furnished' else 0,
                'furnishingstatus_semi-furnished': 1 if request.POST.get('furnishingstatus') == 'semi-furnished' else 0,
                'furnishingstatus_unfurnished': 1 if request.POST.get('furnishingstatus') == 'unfurnished' else 0,
            }

            df = pd.DataFrame([input_data])
            from .model_train import engineer_features
            df_processed = engineer_features(df, is_training=False)
            df_processed = df_processed[feature_names]
            X_scaled = scaler.transform(df_processed)
            svm_pred = svm_model.predict(X_scaled)[0]
            dt_pred = dt_model.predict(df_processed)[0]
            svm_pred_original = np.expm1(svm_pred)
            dt_pred_original = np.expm1(dt_pred)
            
            return render(request, 'predict.html', {
                'prediction_svm': f'Rs. {svm_pred_original:,.2f}',
                'prediction_dt': f'Rs. {dt_pred_original:,.2f}',
                'form_data': request.POST,
                'encoder_mappings': encoder_mappings
            })

        return render(request, 'predict.html', {
            'encoder_mappings': encoder_mappings
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")  
        return render(request, 'predict.html', {
            'error': f"An error occurred: {str(e)}",
            'encoder_mappings': encoder_mappings
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
