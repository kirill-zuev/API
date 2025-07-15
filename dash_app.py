import datetime
import httpx
import dash
import os

import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
from flask import send_from_directory
from dash import dcc, html

app = dash.Dash(__name__)

server = app.server

FASTAPI_URL = "http://0.0.0.0:9009/api/tts"
TIMEOUT = httpx.Timeout(600.0)
LANGUAGES = {
    'Русский': 'ru',
    'Английский': 'en',
    'Китайский': 'zh-cn',
    'Итальянский': 'it',
    'Французский': 'fr',
    'Японский': 'ja',
    'Казахский': 'kaz',
    'Греческий': 'grc',
}
GENERATING = False
AUDIO_DIR = "/home/teslaa2/projects/kp.zuev/voicegen/TTSC/TTSC/bin/Dash/"
SAVED_AUDIO = []
AUDIO_URL = [None]

def extract_time(file_name):
    return datetime.datetime.strptime(file_name.split('.')[0], "%d-%m-%Y_%H-%M-%S_%f")

def load_audio() -> None:
    existing_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    sorted_files = sorted(existing_files, key=extract_time)
    existing_files = sorted_files.copy()
    for file in existing_files:
        text_path = os.path.join(AUDIO_DIR, f"{file[:-4]}.txt")
        speed_path = os.path.join(AUDIO_DIR, f"speed_{file[:-4]}.txt")
        text = "-"
        speed = "-"
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as text_file:
                text = text_file.read()
        if os.path.exists(speed_path):
            with open(speed_path, "r", encoding="utf-8") as text_file:
                speed = text_file.read()
        SAVED_AUDIO.append({"text": text, "filename": file, "speed": speed})
load_audio()

def text_to_audio(data, lang, speed):
    request_data = {
        "text": data
    }
    params = {
        "voice": False,
        "lang": lang,
        "speed": speed,
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
            with open(f"./Dash/speed_{audio[:-4]}.txt", "w", encoding="utf-8") as speed_file:
                speed_file.write(str(speed))
            return audio
        else:
            print(f"{response.status_code}")
            return None

sidebar = html.Div(
    [
        html.H2("Оглавление", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Генерация", href="/", active="exact", style={"display": "block"}),
                dbc.NavLink("Инструкция", href="/page", active="exact", style={"display": "block"}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style= {"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "5rem 1rem", "background-color": "#f8f9fa"},
)

content = html.Div([
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

    html.Div([
        html.Audio(id='audio-player', controls=True, style={'margin-top': '40px', 'width': '15%'}),
        html.Div([
            dcc.Slider(
                id='speed-slider',
                min=0,
                max=1.99,
                step=0.01,
                value=1,
                marks={0: '0%', 1: '^', 1.99: '100%'},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={'margin-top': '60px', 'width': '15%', 'margin-left': '10px'}),
    ], style={'display': 'flex', 'gap': '10px'}),

    html.Div([
        html.H3("Результаты"),
        html.Table([
            html.Thead(html.Tr([html.Th("Текст"), html.Th("Аудио")]))
        ], id='audio-table', style={'border': '1px solid black', 'border-collapse': 'collapse'})
        # html.Ul(id='audio-list')
    ])
])

page_content = html.Div([
    html.H1("Инструкция"),
    html.H4("Ударения"),
    html.P("Для регулирования ударений в словах используйте служебный символ {+}.  Символ должен следовать после ударной гласной. "),
    html.H4("Паузы"),
    html.P("В тексте можно использовать запятые для регулирования пауз между словами и фразами."),
    html.H4("Допустимые символы"),
    html.P("В тексте, подаваемом для синтеза, не стоит использовать служебные символы кроме: < >{ }[] - ( ) : ; №."),
    html.H4("Использование числительных"),
    html.P("Модель способна воспринимать числительные, однако при передаче дат следует придерживаться формата День.Месяц.Год (например, 18.05.2025). Во избежание ошибок, в случае использования числительных в нестандартных ситуациях, желательно записывать их в форме слов."),
    html.H4("Смешение языков и алфавитов"),
    html.P("Произношение модели фиксируется выбором языка. В случае каждого из доступных языков модель ориентируется на соответствующий алфавит. В случае присутствия в тексте (допустим, русскоязычном) слова из другого языка, записанного с помощью другого алфавита (наиболее встречающимся примером являются англоязычные имена собственные, например, названия авиакомпаний), он будет произнесен с соответствующим акцентом. В виду этого, для более естественного звучания рекомендуется транслитерировать иностранные слова (например, Southwind Airlines заменится на Саузвинд Эйрлайнс). Важное исключение! Названия терминалов должны быть всегда написаны латиницей, а «Шереметьево» - кириллицей."),
    html.H4("Максимальное число символов в предложении"),
    html.P("Модель обрабатывает текст, самостоятельно разбивая его на них. Максимальная длина одного предложения – 200 символов. Превышение этого лимита может привести к появлению звуковых артефактов. Слишком длинное предложение можно разделить, просто поставив дополнительную точку в месте, где присутствует интонационная пауза."),
])

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    html.Div(id="page-content", style={"margin-left": "18rem", "padding": "5rem"})
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
    ],
    [
        State('text-input', 'value'),
        State('language-dropdown', 'value'),
        State('speed-slider', 'value'),
    ],
)
def generate_audio(n_clicks, text, lang, speed):
    if n_clicks > 0 and text:
        audio_file = text_to_audio(text, lang, speed)

        AUDIO_URL[0] = f"/audio/{audio_file}"
        SAVED_AUDIO.append({"text": text, "filename": audio_file, "speed": str(speed)})

        audio_list = [
            html.Tr([
                html.Td(f"{i + 1}.", style={'width': '1%', 'border': '1px solid black'}),
                html.Td(item['text'], style={'width': '60%', 'border': '1px solid black', 'padding-left': '5px'}),
                html.Td(html.Audio(src=f"/audio/{item['filename']}", controls=True, style={'width': '50%', 'margin-top': '10px', 'padding-left': '40px'}), style={'width': '40%', 'border': '1px solid black'}),
                html.Td("Скорость: "+item['speed'], style={'width': '9%', 'border': '1px solid black', 'padding-left': '5px'}),
            ]) 
            for i, item in enumerate(SAVED_AUDIO)
        ]

        return AUDIO_URL[0], audio_list, None, 0
    else:
        audio_list = [
            html.Tr([
                html.Td(f"{i + 1}.", style={'width': '1%', 'border': '1px solid black'}),
                html.Td(item['text'], style={'width': '60%', 'border': '1px solid black', 'padding-left': '5px'}),
                html.Td(html.Audio(src=f"/audio/{item['filename']}", controls=True, style={'width': '40%', 'margin-top': '10px', 'padding-left': '40px'}), style={'width': '40%', 'border': '1px solid black'}),
                html.Td("Скорость: "+item['speed'], style={'width': '9%', 'border': '1px solid black', 'padding-left': '5px'}),
            ])
            for i, item in enumerate(SAVED_AUDIO)
        ]
        return AUDIO_URL[0], audio_list, None, 0

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == "/":
        return content
    elif pathname == "/page":
        return page_content
    else:
        return html.H1("404: Page not found")


@app.server.route('/audio/<filename>')
def server_audio(filename):
    return send_from_directory("/home/teslaa2/projects/kp.zuev/voicegen/TTSC/TTSC/bin/Dash/", filename)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=9005)