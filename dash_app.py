import torch
import httpx
import asyncio
import os
import datetime
from synthesize_new import main_tts

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import send_from_directory

app = dash.Dash(__name__)

server = app.server

FASTAPI_URL = "http://10.96.75.253:9001/api/tts"
TIMEOUT = httpx.Timeout(600.0)
LANGUAGES = {
    'Русский': 'ru',
    'Английский': 'en',
    'Китайский': 'zh-cn',
    'Итальянский': 'it',
    'Французский': 'fr',
    'Японский': 'ja',
}
GENERATING = False
AUDIO_DIR = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/TTSC/bin/Dash/"
SAVED_AUDIO = []
AUDIO_URL = [None]

def extract_time(file_name):
        return datetime.datetime.strptime(file_name.split('.')[0], "%d-%m-%Y_%H-%M-%S_%f")

def load_audio():
    existing_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    sorted_files = sorted(existing_files, key=extract_time)
    existing_files = sorted_files.copy()
    for file in existing_files:
        text_path = os.path.join(AUDIO_DIR, f"{file[:-4]}.txt")
        text = "-"
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as text_file:
                text = text_file.read()
        SAVED_AUDIO.append({"text": text, "filename": file})
load_audio()

# def text_to_audio(data, lang):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     file_path = ""

#     if not os.path.exists(file_path):
#         file_path = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3"
#         if lang == "en":
#             file_path = "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-EN.mp3"

#     time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S_%f")
#     audio = f"{time}.wav"
#     if lang == "en":
#         main_tts(data, "tts_models/multilingual/multi-dataset/xtts_v2", f"./Dash/{audio}", file_path, lang, device)
#     else:
#         main_tts(data, "tts_models/multilingual/multi-dataset/xtts_v2", f"./Dash/{audio}", file_path, lang, device)
#     saved_audio.append({"text": data, "filename": audio})
#     with open(f"./Dash/{audio[:-4]}.txt", "w", encoding="utf-8") as text_file:
#         text_file.write(data)
#     return audio

def text_to_audio(data, lang):
    request_data = {
        "text": data
    }
    params = {
        "voice": False,
        "lang": lang
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(FASTAPI_URL, json=request_data, params=params)
        if response.status_code == 200:
            audio_bytes = response.content
            time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S_%f")
            audio = f"{time}.wav"
            with open(f"./Dash/{audio}", "wb") as audio_file:
                audio_file.write(audio_bytes)
            with open(f"./Dash/{audio[:-4]}.txt", "w", encoding="utf-8") as text_file:
                text_file.write(data)
            return audio
        else:
            print(f"{response.status_code}")
            return None

app.layout = html.Div([
    html.H1("Генерация аудио из текста"),

     html.Div([
        dcc.Textarea(
            id='text-input',
            placeholder='Введите текст...',
            style={'width': '60%', 'height': 100, 'margin-right': '40px'}
        ),

        html.Div([
            dcc.Dropdown(
                id='language-dropdown',
                options=[{'label': lang, 'value': code} for lang, code in LANGUAGES.items()],
                value='ru',
                style={'width': '100%', 'height': 45, 'margin-bottom': '10px'}
            ),

            dcc.Loading(
                id="loading",
                children=[html.Button("Генерировать аудио", id="generate-button", n_clicks=0, style={'height': 45, 'width': '100%'}),
                          html.Div(id="loading-output")],
                type="circle"),

        ], style={'display': 'flex', 'flexDirection': 'column', 'width': '40%'})

    ], style={'display': 'flex', 'alignItems': 'flex-start'}),

    html.Audio(id='audio-player', controls=True, style={'margin-top': '40px'}),

    html.Div([
        html.H3("Результаты"),
        html.Table([
            html.Thead(html.Tr([html.Th("Текст"), html.Th("Аудио")]))
        ], id='audio-table', style={'border': '1px solid black', 'border-collapse': 'collapse'})
        # html.Ul(id='audio-list')
    ])
])

@app.callback(
    [
        Output('audio-player', 'src'),
        Output('audio-table', 'children'),
        Output('loading-output', 'children'),
        Output('generate-button', 'n_clicks'),
    ],
    [
        Input('generate-button', 'n_clicks'),
        Input('text-input', 'value'),
        Input('language-dropdown', 'value'),
    ]
)
def generate_audio(n_clicks, text, lang):
    if n_clicks > 0 and text:
        audio_file = text_to_audio(text, lang)

        AUDIO_URL[0] = f"/audio/{audio_file}"
        SAVED_AUDIO.append({"text": text, "filename": audio_file})

        audio_list = [
            html.Tr([
                html.Td(f"{i + 1}.", style={'width': '1%', 'border': '1px solid black'}),
                html.Td(item['text'], style={'width': '59%', 'border': '1px solid black', 'padding-left': '5px'}),
                html.Td(html.Audio(src=f"/audio/{item['filename']}", controls=True, style={'width': '40%', 'margin-top': '10px', 'padding-left': '40px'}))
            ]) 
            for i, item in enumerate(SAVED_AUDIO)
        ]

        return AUDIO_URL[0], audio_list, None, 0
    else:
        audio_list = [
            html.Tr([
                html.Td(f"{i + 1}.", style={'width': '1%', 'border': '1px solid black'}),
                html.Td(item['text'], style={'width': '59%', 'border': '1px solid black', 'padding-left': '5px'}),
                html.Td(html.Audio(src=f"/audio/{item['filename']}", controls=True, style={'width': '40%', 'margin-top': '10px', 'padding-left': '40px'}))
            ])
            for i, item in enumerate(SAVED_AUDIO)
        ]
        return AUDIO_URL[0], audio_list, None, 0

@app.server.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory("/home/ubuntu/projects/kp.zuev/voicegen/TTSC/TTSC/bin/Dash/", filename)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=9005)