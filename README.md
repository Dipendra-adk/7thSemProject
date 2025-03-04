# House Price Prediction System

This project is a House Price Prediction System developed to estimate property prices in Nepal using Decision Tree and Support Vector Machine (SVM) algorithms. Leveraging historical property data, the system provides accurate price predictions based on user-inputted property details.

## Features

- **Accurate Predictions:** Utilizes Decision Tree and SVM algorithms to provide reliable house price estimates.
- **User-Friendly Interface:** Allows users to input property details effortlessly and receive instant predictions.
- **Secure Authentication:** Ensures user data protection through integrated authentication mechanisms.
- Facilitates direct communication between buyers and sellers:
  - Buyers can contact sellers by filling out a contact form with their message.
  - Upon submission, an email is sent directly to the seller via the host user.
- Automated notifications for property listings:
  - Sellers receive an email notification when their listed properties are approved or declined by the admin.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Dipendra-adk/7thSemProject.git
2. **Navigate to the Project Directory:**
   ```bash
   cd 7thSemProject/homePricePredictior
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. Set Up the Database:
   * Ensure PostgreSQL is installed and running.
   * Create a database named homedb.
     ```bash
     DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.postgresql',
           'NAME': 'homedb',
           'USER': 'postgres',
           'PASSWORD': 'admin',
           'HOST': 'localhost',
           'PORT': '',           
       }
     }
   * Apply migrations:
     ```bash
     python manage.py migrate
5. Run the Development Server:
   ```bash
   python manage.py runserver

## Usage
* Register or Log In: Create an account or log in with existing credentials.
* Input Property Details: Enter information such as area in aana, number of floors, road width (in feet), city, and road type.
* Get Predictions: Receive estimated house prices based on the provided details.
* Compare Results: View and compare predictions from both Decision Tree and SVM models.

## Screenshots
### 1. Login/Signup
<img src="./homePricePredictior/ss/login_signup.png" alt="home" width="500" />

### 2. Home Page
<img src="./homePricePredictior/ss/home.png" alt="home" width="500" />

### 3. Listing properties for buying
<img src="./homePricePredictior/ss/properties.png" alt="properties" width="500" />

### 4. Adding properties detail for selling
<img src="./homePricePredictior/ss/sell_properties.png" alt="selling details" width="500" />

### 5. contact
<img src="./homePricePredictior/ss/contact.png" alt="contact" width="500" />

### 6. User Dashboard
<img src="./homePricePredictior/ss/user_dashboard.png" alt="User Dashboard" width="500" />

### 7. Admin Dashboard
<img src="./homePricePredictior/ss/admin_dashboard.png" alt="Admin Dashboard" width="500" />

### 8. House Price Prediction
<img src="./homePricePredictior/ss/price_prediction.png" alt="House Price Prediction" width="500" />

### 9. Price Prediction result with both models
<img src="./homePricePredictior/ss/prediction_result.png" alt="Prediction" width="500" />

## Technologies Used
* Backend: Django
* Frontend: HTML, CSS, JavaScript
* Database: PostgreSQL
* Machine Learning: scikit-learn

## Contributors
   * [Dipendra Adhikari](https://github.com/Dipendra-adk)
   * [Pursottam Bhandari](https://github.com/pursottam1234)
   * [Phanendra Jaisi](https://github.com/phanindraspk)
