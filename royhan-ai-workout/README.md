# AI Workout Analyzer

##  Project Overview (Big Picture)
This project is an AI-based system that analyzes a user's physical condition and recommends an appropriate workout level: Rest, Light, or Heavy.

The goal is to help users avoid overtraining and improve workout efficiency using simple health indicators.

---

##  Data Collection (Get the Data)
The system uses user input data such as:

- Sleep hours
- Energy level (1–10)
- Muscle soreness (1–10)
- Stress level (1–10)
- Consecutive workout days
- Weight, Height, Gender, Body fat

Example:
```python
data = {
    "sleep_hours": 5,
    "energy_level": 4,
    "muscle_soreness": 7,
    "stress_level": 6,
    "consecutive_workout_days": 3,
    "weight": 55,
    "height": 165,
    "gender": "female",
    "body_fat": 24
}
##  Data Exploration (Explore Data)

Key insights:
- Low sleep → body recovery is low  
- High muscle soreness → muscles need rest  
- High stress → avoid heavy workouts  

---

##  Data Preparation

The input data is structured into a dictionary format for processing by the AI model.

---

##  Model & Processing

This project uses a generative AI model together with a rule-based scoring system.

- The AI model generates recommendations based on user input
- A fallback scoring system ensures the system can still function without API access

The system evaluates:
- Recovery condition
- Physical readiness
- Risk of injury 

##  Model Improvement (Fine-tuning)

The scoring logic was adjusted to improve accuracy.

For example:
- High muscle soreness significantly reduces the score
- Low sleep (less than 5 hours) strongly decreases workout readiness
- High stress reduces the intensity recommendation

These adjustments help the system better reflect real human recovery conditions.

---

##  Results (3 Cases)

### 🟥 Case 1: Rest
**Input:**
- Sleep: 3  
- Energy: 2  
- Soreness: 8  
- Stress: 8  

**Output:**
- Decision: Rest  
- Recommendation: Full rest, hydration, sleep recovery  

---

### 🟨 Case 2: Light Workout
**Input:**
- Sleep: 5  
- Energy: 4  
- Soreness: 7  
- Stress: 6  

**Output:**
- Decision: Light Workout  
- Recommendation: Light cardio, stretching  

---

### 🟩 Case 3: Heavy Workout
**Input:**
- Sleep: 8  
- Energy: 9  
- Soreness: 2  
- Stress: 2  

**Output:**
- Decision: Heavy Workout  
- Recommendation: Strength training, high intensity workout  

---

##  Conclusion

The AI system can effectively classify workout readiness based on simple health indicators.

---

##  Deployment

Currently, the system runs in Jupyter Notebook.

It can be deployed in the future as:
- A web application
- A mobile fitness assistant
- Integration with wearable devices

##  Future Improvements

- Develop a mobile application  
- Connect with wearable devices (e.g., smartwatch)  
- Improve model accuracy with real datasets  
