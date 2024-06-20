from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
import pandas as pd

# Load your trained model
model = load_model('food_CNN_ep80.h5')

# Define the class labels and corresponding calories
classes = {
    'Dosa': 168,
    'Gulab Jamun': 97,
    'Idly': 61,
    'Rice': 130,
    'Vada': 97,
    'Vada_pav': 197,
    'french_fries': 312,
    'pizza': 266,
}

# Load the cookbook data
cookbook = pd.read_csv('static/Cook_Book_1.csv')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction_result = None
    calories = None
    recipe = None
    if request.method == 'POST':
        file = request.files['file']
        img = image.load_img(BytesIO(file.read()), target_size=(64, 64))  # Adjust target_size based on your model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image (rescale, etc.) if necessary
        img_array = img_array / 255.0  # Assuming rescaling was done during training

        # Predict the class of the image
        prediction = model.predict(img_array)

        # Get the predicted class
        prediction_result = list(classes.keys())[np.argmax(prediction)]

        # Get the calories for the predicted dish
        calories = classes[prediction_result]

        # Get the recipe for the predicted dish
        predicted_dish = cookbook[cookbook['Dish'] == prediction_result]
        if not predicted_dish.empty:
            recipe = predicted_dish['Recipe'].values[0]
        else:   
            recipe = "Recipe not found in cookbook"

    return render_template('upload.html', prediction_result=prediction_result, calories=calories, recipe=recipe)

if __name__ == '__main__':
    app.run(debug=True)
