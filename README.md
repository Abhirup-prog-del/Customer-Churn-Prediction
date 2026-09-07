Key Project Features
Interactive UI: Built with Streamlit for a responsive and user-friendly experience.
Machine Learning Model: Utilizes a Random Forest Classifier trained on a telecom churn dataset to achieve high prediction accuracy.
Dynamic Visualizations: Employs Plotly to create an intuitive gauge chart for visualizing churn probability.
Model Explainability: Shows the key factors that influence each prediction, providing actionable insights instead of just a number.
Explanation of Model Features
Below is a brief description of the customer data used by the model to make predictions:

Demographics: Gender, SeniorCitizen, Partner, Dependents (whether the customer has a partner or dependents).
Account Information:
Tenure: How long the customer has been with the company (in months).
Contract: The customer's contract term (Month-to-month, One year, Two year).
PaymentMethod: How the customer pays their bills.
PaperlessBilling: Whether the customer uses paperless billing.
MonthlyCharges & TotalCharges: The amount the customer is charged.
Service Information: Details about the services the customer has subscribed to, such as PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
