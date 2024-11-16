import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# Load the TensorFlow model
model = tf.keras.models.load_model("C:/Users/lenovo/OneDrive/Desktop/Image detection/model.keras")

# List of class names for predictions
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Function to make predictions
def model_prediction(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((64, 64))  # Resize image to match model input
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Plant Disease Recognition System 🌿🔍", className="text-center my-4"))
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-image-upload')
        ])
    ]),

    dbc.Row([
        dbc.Col(html.H4(id='prediction-result', className="text-center my-4"))
    ])
], fluid=True)

# Callback to process the uploaded image and make predictions
@app.callback(
    [Output('output-image-upload', 'children'), Output('prediction-result', 'children')],
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def update_output(image_content):
    if image_content is not None:
        # Decode the base64 image
        content_type, content_string = image_content.split(',')
        decoded = base64.b64decode(content_string)
        
        # Display the image
        image = Image.open(io.BytesIO(decoded))
        image_html = html.Img(src=image_content, style={'max-width': '100%', 'height': 'auto'})

        # Make prediction
        prediction_index = model_prediction(decoded)
        predicted_class = class_names[prediction_index]

        # Display prediction result
        result_text = f"Model is predicting: {predicted_class}"
        
        return image_html, result_text

    return None, "No image uploaded"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
