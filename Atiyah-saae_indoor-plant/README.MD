# Indoor Plant Growing Recommendation AI

This project helps users grow plants indoors or in small spaces such as dorm rooms or apartments.

The user enters the name of a plant they want to grow, and the AI system will recommend the best way to grow it.

## Features
- Recommend the best location to place the plant in a room
- Suggest the sunlight level needed for the plant
- Suggest watering frequency
- Provide simple plant care tips

## Example

If a user wants to grow Chili, the AI may recommend:

Location: Balcony or near the window  
Sunlight: High sunlight  
Watering: Water every day or when the soil is dry  
Tip: Chili plants grow best with strong sunlight.

## Dataset

The dataset contains information about plants including:

- Plant name
- Best location to grow
- Sunlight requirement
- Watering frequency
- Plant care tips

## Technologies Used

- Python
- Pandas
- Jupyter Notebook

## Goal of the Project

The goal of this project is to create a simple AI system that helps beginners grow plants easily, especially in small indoor spaces.

## How It Works

The system uses a simple rule-based scoring method to match plant requirements with the user's room conditions.

It evaluates:
- Sunlight compatibility
- Space compatibility

Each factor is given a score, and the total score determines how suitable the plant is for the user's room.

## Limitations

- The dataset is small and limited to basic plant types
- The system uses rule-based logic instead of advanced machine learning models
- Recommendations may not cover all real-world conditions

## Future Improvements

- Use machine learning models for better predictions
- Expand the dataset with more plant types
- Develop a graphical user interface (GUI)