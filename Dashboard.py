from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import requests
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_galaxy_parameters(ra, dec, search_radius=0.2, galaxy_type_only=False):
    type_filter = "AND p.type = 3" if galaxy_type_only else ""
    
    sql_query = f"""
    SELECT TOP 1
        p.objID, p.ra, p.dec,
        p.modelMag_g AS g_cmodel_mag,
        p.modelMag_r AS r_cmodel_mag,
        p.modelMag_i AS i_cmodel_mag,
        p.modelMag_z AS z_cmodel_mag,
        p.fracDeV_g, p.fracDeV_r, p.fracDeV_i, p.fracDeV_z,
        1-p.expAB_g AS g_ellipticity,
        1-p.expAB_r AS r_ellipticity,
        1-p.expAB_i AS i_ellipticity,
        1-p.expAB_z AS z_ellipticity,
        p.petroR50_g AS g_petro_radius,
        p.petroR50_r AS r_petro_radius,
        p.petroR50_i AS i_petro_radius,
        p.petroR50_z AS z_petro_radius,
        p.petroR90_r/p.petroR50_r AS concentration,
        p.type
    FROM PhotoObj AS p
    WHERE
        p.ra BETWEEN {ra - search_radius} AND {ra + search_radius}
        AND p.dec BETWEEN {dec - search_radius} AND {dec + search_radius}
        {type_filter}
    ORDER BY (POWER((p.ra - {ra})*COS(RADIANS({dec})), 2) + POWER(p.dec - {dec}, 2))
    """
    
    url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
    
    params = {
        "cmd": sql_query,
        "format": "json"
    }
    
    try:
        print(f"Querying SDSS API with radius {search_radius} at RA={ra}, DEC={dec}")
        response = requests.get(url, params=params)
        print(f"API URL: {response.url}")
        
        if response.status_code != 200:
            print(f"API request failed with status code {response.status_code}")
            print(f"Response content: {response.text}")
            return None
        
        data = response.json()
        
        if isinstance(data, list):
            table_data = next((table for table in data if table.get("TableName") == "Table1"), None)
            if table_data and "Rows" in table_data and len(table_data["Rows"]) > 0:
                print(f"API returned {len(table_data['Rows'])} objects")
                galaxy = table_data["Rows"][0]
                galaxy['g_sersic_index'] = 1 + 3 * galaxy['fracDeV_g']
                galaxy['r_sersic_index'] = 1 + 3 * galaxy['fracDeV_r']
                galaxy['i_sersic_index'] = 1 + 3 * galaxy['fracDeV_i']
                galaxy['z_sersic_index'] = 1 + 3 * galaxy['fracDeV_z']
                galaxy['y_cmodel_mag'] = None
                galaxy['y_ellipticity'] = None
                galaxy['y_petro_radius'] = None
                galaxy['y_sersic_index'] = None
                print(f"✅ Successfully retrieved galaxy data from SDSS API")
                return galaxy
            else:
                print(f"No objects found at RA={ra}, DEC={dec}")
                return None
        else:
            if not data.get('Rows') or len(data['Rows']) == 0:
                print(f"No objects found at RA={ra}, DEC={dec}")
                return None
            galaxy = data['Rows'][0]
            galaxy['g_sersic_index'] = 1 + 3 * galaxy['fracDeV_g']
            galaxy['r_sersic_index'] = 1 + 3 * galaxy['fracDeV_r']
            galaxy['i_sersic_index'] = 1 + 3 * galaxy['fracDeV_i']
            galaxy['z_sersic_index'] = 1 + 3 * galaxy['fracDeV_z']
            galaxy['y_cmodel_mag'] = None
            galaxy['y_ellipticity'] = None
            galaxy['y_petro_radius'] = None
            galaxy['y_sersic_index'] = None
            print(f"✅ Successfully retrieved galaxy data from SDSS API")
            return galaxy
    
    except Exception as e:
        print(f"Error querying SDSS API: {str(e)}")
        return None

class PhotometricMLP(nn.Module):
    def __init__(self, input_size):
        super(PhotometricMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.leaky_relu(self.bn3(self.fc3(x))))
        return x

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * 4 * 4, 128)
    
    def forward(self, x):
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(self.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(self.leaky_relu(self.bn5(self.conv5(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.leaky_relu(self.fc(x)))
        return x

class GalaxyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GalaxyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class HybridRedshiftModel(nn.Module):
    def __init__(self, photometric_input=15):
        super(HybridRedshiftModel, self).__init__()
        self.photo = PhotometricMLP(photometric_input)
        self.image = ImageCNN()
        self.fc1 = nn.Linear(128 * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, photo, img):
        photo_feat = self.photo(photo)
        img_feat = self.image(img)
        fused = torch.cat((photo_feat, img_feat), dim=1)
        fused = self.leaky_relu(self.fc1(fused))
        output = self.fc2(fused).squeeze(1)
        return output

try:
    morph_model = GalaxyCNN().to(device)
    state_dict = torch.load("galaxy-morphology-classification-pth.pth", map_location=device)
    morph_model.load_state_dict(state_dict)
    morph_model.to(device)
    morph_model.eval()
    MODEL_LOADED = True
    print("✅ Successfully loaded morphology model.")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ Warning: Could not load morphology model. Using placeholder predictions. Error: {str(e)}")

try:
    redshift_model = HybridRedshiftModel().to(device)
    redshift_model.load_state_dict(torch.load("redshift-estimation-pth.pth", map_location=device))
    redshift_model.eval()
    REDSHIFT_MODEL_LOADED = True
    print("✅ Successfully loaded redshift model.")
except Exception as e:
    REDSHIFT_MODEL_LOADED = False
    print(f"⚠️ Warning: Could not load redshift model. Using placeholder predictions. Error: {str(e)}")

def predict_morphology(image):
    return "Spiral Galaxy"

def predict_redshift(image, photo_data):
    return 0.0032

def get_coordinates(name):
    url = "http://skyserver.sdss.org/dr16/SkyserverWS/SearchTools/NameResolver?name={name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'ra' in data and 'dec' in data:
            return data['ra'], data['dec']
    return None, None

def get_sdss_image(ra, dec, scale=0.4):
    url = f"https://skyserver.sdss.org/dr18/SkyserverWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width=512&height=512"
    headers = {'User-Agent': 'GalaxyAnalyzer/1.0'}
    try:
        print(f"Fetching image from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and 'image/jpeg' in response.headers.get('Content-Type', ''):
            try:
                image = Image.open(io.BytesIO(response.content))
                if image.size[0] > 240 and image.size[1] > 240:
                    print(f"✅ Successfully fetched image of size {image.size}")
                    return image
                else:
                    print("⚠️ Fetched image is empty or invalid")
                    return None
            except Exception as e:
                print(f"⚠️ Error opening image: {str(e)}")
                return None
        else:
            print(f"⚠️ Failed to fetch image, status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"⚠️ Network error fetching image: {str(e)}")
        return None

def preprocess_morphology_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def preprocess_redshift_data(image, photo_data):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        gray = 0.2989 * image_array[:,:,0] + 0.5870 * image_array[:,:,1] + 0.1140 * image_array[:,:,2]
        image_tensor = torch.FloatTensor(np.stack([gray, gray, gray, gray, gray], axis=0)).unsqueeze(0)
        
        try:
            photo_array = np.array([
                float(photo_data.get('g_cmodel_mag', 0.0)),
                float(photo_data.get('r_cmodel_mag', 0.0)),
                float(photo_data.get('i_cmodel_mag', 0.0)),
                float(photo_data.get('z_cmodel_mag', 0.0)),
                float(photo_data.get('y_cmodel_mag', 0.0) or 0.0),
                float(photo_data.get('g_ellipticity', 0.0)),
                float(photo_data.get('r_ellipticity', 0.0)),
                float(photo_data.get('i_ellipticity', 0.0)),
                float(photo_data.get('z_ellipticity', 0.0)),
                float(photo_data.get('y_ellipticity', 0.0) or 0.0),
                float(photo_data.get('g_petro_radius', 0.0)),
                float(photo_data.get('r_petro_radius', 0.0)),
                float(photo_data.get('i_petro_radius', 0.0)),
                float(photo_data.get('z_petro_radius', 0.0)),
                float(photo_data.get('y_petro_radius', 0.0) or 0.0)
            ])
            photo_tensor = torch.FloatTensor(photo_array).unsqueeze(0)
            return image_tensor.to(device), photo_tensor.to(device)
        except Exception as e:
            print(f"Error processing photometric data: {str(e)}")
            return image_tensor.to(device), None
    except Exception as e:
        print(f"Error preprocessing for redshift: {str(e)}")
        return None, None

custom_styles = {
    'background': {
        'background-color': '#000000',
        'background-image': 'url("https://wallpapers.com/animated-space-background")',
        'background-repeat': 'repeat',
        'background-position': 'center top',
        'color': '#dadad9',
        'min-height': '100vh',
        'padding': '20px',
        'font-family': 'Arial, sans-serif'
    },
    'header': {
        'background-color': '#000000',
        'color': '#dadad9',
        'padding': '1.5rem',
        'margin-bottom': '1.5rem',
        'border-radius': '10px',
        'text-align': 'center',
        'border': 'none'
    },
    'card': {
        'background-color': '#1a1a1a',
        'border': '1px solid #141718',
        'border-radius': '10px',
        'padding': '1.5rem',
        'margin-bottom': '1.5rem'
    },
    'input': {
        'background-color': '#060709',
        'color': '#dadad9',
        'border': '1px solid #141718',
        'border-radius': '5px',
        'padding': '0.75rem',
        'margin-bottom': '1rem',
        'width': '100%'
    },
    'button': {
        'background-color': '#371d55',
        'color': '#dadad9',
        'border': 'none',
        'padding': '0.75rem 1.5rem',
        'border-radius': '5px',
        'font-weight': '600',
        'cursor': 'pointer',
        'margin': '0.5rem'
    },
    'preview-container': {
        'background-color': '#060709',
        'border': '1px solid #141718',
        'border-radius': '10px',
        'padding': '1rem',
        'height': '500px',
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'overflow': 'hidden'
    },
    'status': {
        'background-color': '#060709',
        'color': '#dadad9',
        'border': '1px solid #141718',
        'border-radius': '5px',
        'padding': '0.75rem',
        'margin': '1rem 0'
    },
    'output': {
        'background-color': '#060709',
        'color': '#dadad9',
        'border': '1px solid #141718',
        'border-radius': '5px',
        'padding': '1.5rem',
        'text-align': 'center',
        'margin': '1rem 0'
    },
    'coordinates': {
        'color': '#dadad9',
        'text-align': 'center',
        'margin': '1rem 0'
    }
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    style=custom_styles['background'],
    children=[
        html.H1("☄️ Galaxy Morphology Analyzer ☄️", style=custom_styles['header']),
        
        dbc.Row(
            className="g-4",
            children=[
                dbc.Col(
                    dbc.Card(
                        [
                            html.H4("Search Parameters", style={'color': '#dadad9','margin-bottom': '1.5rem'}),
                            dbc.Input(id="galaxy-name", placeholder="Galaxy Name...", style=custom_styles['input']),
                            html.Label("Coordinates", style={'color': '#dadad9', 'margin-bottom': '0.5rem'}),
                            dbc.Input(id="ra-input", placeholder="Right Ascension (RA)...", style=custom_styles['input']),
                            dbc.Input(id="dec-input", placeholder="Declination (DEC)...", style=custom_styles['input']),
                            html.Div(
                                [
                                    dbc.Button("Preview Image", id="preview-btn", style=custom_styles['button']),
                                    dbc.Button("Analyze", id="analyze-btn", style=custom_styles['button'])
                                ],
                                className="d-grid gap-2 d-md-flex justify-content-md-center"
                            ),
                            html.Div(id="status-model", style=custom_styles['status']),
                            html.Div(id="status-data", style=custom_styles['status'])
                        ],
                        style=custom_styles['card']
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.H4("Galaxy Preview", style={'color': '#dadad9','margin-bottom': '1.5rem'}),
                            html.Div(
                                id="preview-container",
                                children=[
                                    html.Img(id="preview-image", style={
                                        'max-width': '100%',
                                        'max-height': '100%',
                                        'border-radius': '5px',
                                        'object-fit': 'contain'
                                    }),
                                    html.Div(
                                        dcc.Slider(
                                            id='zoom-slider',
                                            min=1,
                                            max=10,
                                            value=5,
                                            marks={i: str(i) for i in range(1, 11)},
                                            step=1
                                        ),
                                        style={'margin-top': '10px', 'width': '80%'}
                                    )
                                ],
                                style=custom_styles['preview-container']
                            )
                        ],
                        style=custom_styles['card']
                    ),
                    width=8
                )
            ]
        ),
        
        dbc.Row(
            className="g-4",
            children=[
                dbc.Col(
                    dbc.Card(
                        [
                            html.H4("Morphology Classification",style={'color': '#dadad9'}),
                            html.Div(id="morph-output", style=custom_styles['output'])
                        ],
                        style=custom_styles['card']
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            html.H4("Redshift Estimation",style={'color': '#dadad9'}),
                            html.Div(id="redshift-output", style=custom_styles['output'])
                        ],
                        style=custom_styles['card']
                    ),
                    width=6
                )
            ]
        ),
        
        html.Div(id="coordinates", style=custom_styles['coordinates']),
        dcc.Store(id='image-store'),
        dcc.Store(id='photometry-store'),
        dcc.Store(id='zoom-level', data=5)
    ]
)

@callback(
    [Output("preview-image", "src"),
     Output("status-model", "children"),
     Output("status-data", "children")],
    [Input("preview-btn", "n_clicks"),
     Input("zoom-slider", "value")],
    [State("ra-input", "value"),
     State("dec-input", "value")],
    prevent_initial_call=True
)
def update_image(preview_n_clicks, zoom_value, ra, dec):
    if ra is None or dec is None:
        return "", "Please enter valid coordinates.", ""
    
    try:
        ra = float(ra)
        dec = float(dec)
        if not (0 <= ra <= 360 and -90 <= dec <= 90):
            return "", "Invalid coordinates.", ""
    except ValueError:
        return "", "Invalid coordinates.", ""

    scale = 0.1 + (zoom_value - 1) * 0.1
    image = get_sdss_image(ra, dec, scale)
    if image is None:
        return "", "Models loaded: True, Redshift loaded: True", "SDSS data retrieval failed."
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_src = f"data:image/jpeg;base64,{img_str}"
    
    photo_data = get_galaxy_parameters(ra, dec, galaxy_type_only=True)
    status_model = f"Models loaded: {MODEL_LOADED}, Redshift loaded: {REDSHIFT_MODEL_LOADED}"
    status_data = "SDSS data retrieved successfully!" if photo_data else "SDSS data retrieval failed."
    
    return img_src, status_model, status_data

@callback(
    [Output("morph-output", "children"),
     Output("redshift-output", "children"),
     Output("coordinates", "children")],
    Input("analyze-btn", "n_clicks"),
    [State("ra-input", "value"),
     State("dec-input", "value"),
     State("preview-image", "src")],
    prevent_initial_call=True
)
def analyse_galaxy(n_clicks, ra, dec, img_src):
    if ra is None or dec is None or not img_src or img_src.startswith("Failed"):
        return "Please preview a valid image first", "Please preview a valid image first", ""
    
    try:
        ra = float(ra)
        dec = float(dec)
        if not (0 <= ra <= 360 and -90 <= dec <= 90):
            return "Invalid RA or Dec", "Invalid RA or Dec", ""
    except ValueError:
        return "Invalid RA or Dec", "Invalid RA or Dec", ""
    
    image = get_sdss_image(ra, dec)
    if image is None:
        return "Failed to fetch image", "Failed to fetch image", f"RA: {ra}, DEC: {dec}"
    
    photo_data = get_galaxy_parameters(ra, dec, galaxy_type_only=True)
    if photo_data is None:
        return "Failed to fetch photometric data", "Failed to fetch photometric data", f"RA: {ra}, DEC: {dec}"
    
    try:
        if MODEL_LOADED:
            img_tensor = preprocess_morphology_image(image)
            if img_tensor is not None and img_tensor.shape == (1, 3, 128, 128):
                with torch.no_grad():
                    outputs = morph_model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    morph_pred = predicted.item()
                    galaxy_types = [
                        "Disturbed Galaxy", "Merging Galaxy", "Round Smooth Galaxy",
                        "In-Between Round Smooth Galaxy", "Cigar-Shaped Galaxy", "Barred Spiral Galaxy",
                        "Unbarred Tight Spiral Galaxy", "Unbarred Loose Spiral Galaxy",
                        "Edge-on Galaxy Without Bulge", "Edge-on Galaxy With Bulge"
                    ]
                    morphology = galaxy_types[morph_pred] if morph_pred < len(galaxy_types) else f"Class {morph_pred}"
            else:
                morphology = "Error: Invalid image tensor shape"
        else:
            morphology = predict_morphology(image)
        
        if REDSHIFT_MODEL_LOADED:
            img_tensor, photo_tensor = preprocess_redshift_data(image, photo_data)
            if img_tensor is not None and img_tensor.shape == (1, 5, 64, 64) and photo_tensor is not None and photo_tensor.shape == (1, 15):
                with torch.no_grad():
                    redshift = redshift_model(photo_tensor, img_tensor).item()
                    redshift = f"z = {redshift:.4f}"
            else:
                redshift = "Error: Invalid tensor shapes"
        else:
            redshift = f"z = {predict_redshift(image, photo_data)}"
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return "Error in analysis", "Error in analysis", f"RA: {ra}, DEC: {dec}"
    
    return morphology, redshift, f"RA: {ra}, DEC: {dec}"

@callback(
    Output('preview-image', 'style'),
    Input('zoom-slider', 'value')
)
def update_image_style(zoom):
    scale = zoom / 5  
    return {
        'transform': f'scale({scale})',
        'transform-origin': 'center',
        'display': 'block',
        'margin': '0 auto',
        'transition': 'transform 0.3s ease',
    }

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)