from TTS.api import TTS
# from TTS.bin.synthesize_new import main_tts
# from TTS.bin.ssml_synthesize import main_tts_ssml
# from synthesize_new import main_tts
# from ssml_synthesize import main_tts_ssml
from IPython.display import Audio
import torch

def prep0(s):
    new = s.replace('<','.')
    new = new.replace('>','.')
    new = new.replace('{','.')
    new = new.replace('}','.')
    new = new.replace('[','.')
    new = new.replace(']','.')
    new = new.replace('(','.')
    new = new.replace(')','.')
    new = new.replace('«','.')
    new = new.replace('»','.')
    new = new.replace('"','.')
    new = new.split()

    return ' '.join(new)

def number_to_words(num, lang, prep_str=False):
    # 0 <= num <= 9999):
    if lang == 'ru' and not prep_str:
        ones = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
        teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
        tens = ["", "десять", "двадцать", "тридцать", "сорок", "пятьдесят",
                "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
        hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
                    "шестьсот", "семьсот", "восемьсот", "девятьсот"]
        thousands = ["", "одна тысяча", "две тысячи", "три тысячи", "четыре тысячи",
                    "пять тысяч", "шесть тысяч", "семь тысяч", "восемь тысяч", "девять тысяч"]
    if lang == 'ru' and prep_str:
        ones = ["нуля", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти"]
        teens = ["десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати",
          "шестнадцати", "семнадцати", "восемнадцати", "девятнадцати"]
        tens = ["", "десяти", "двадцати", "тридцати", "сорока", "пятидесяти",
          "шестидесяти", "семидесяти", "восьмидесяти", "девяносто"]
        hundreds = ["", "ста", "двухсот", "трёхсот", "четырёхсот", "пятисот",
              "шестисот", "семисот", "восьмисот", "девятьсот"]
        thousands = ["", "одной тысячи", "двух тысяч", "трёх тысяч", "четырёх тысяч",
              "пяти тысяч", "шести тысяч", "семи тысяч", "восьми тысяч", "девяти тысяч"]

    if lang == 'en':
        ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
              "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "ten", "twenty", "thirty", "forty", "fifty",
              "sixty", "seventy", "eighty", "ninety"]
        hundreds = ["", "one hundred", "two hundred", "three hundred", "four hundred", "five hundred",
                  "six hundred", "seven hundred", "eight hundred", "nine hundred"]
        thousands = ["", "one thousand", "two thousand", "three thousand", "four thousand",
                  "five thousand", "six thousand", "seven thousand", "eight thousand", "nine thousand"]

    if lang == 'it':
        ones = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]
        teens = ["dieci", "undici", "dodici", "tredici", "quattordici", "quindici",
              "sedici", "diciassette", "diciotto", "diciannove"]
        tens = ["", "dieci", "venti", "trenta", "quaranta", "cinquanta",
              "sessanta", "settanta", "ottanta", "novanta"]
        hundreds = ["", "cento", "duecento", "trecento", "quattrocento", "cinquecento",
                  "seicento", "settecento", "ottocento", "novecento"]
        thousands = ["", "mille", "duemila", "tremila", "quattromila",
                  "cinquemila", "seimila", "settemila", "ottomila", "novemila"]

    if lang == 'fr':
        ones = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
        teens = ["dix", "onze", "douze", "treize", "quatorze", "quinze",
              "seize", "dix-sept", "dix-huit", "dix-neuf"]
        tens = ["", "dix", "vingt", "trente", "quarante", "cinquante",
              "soixante", "soixante-dix", "quatre-vingts", "quatre-vingt-dix"]
        hundreds = ["", "cent", "deux cents", "trois cents", "quatre cents", "cinq cents",
                  "six cents", "sept cents", "huit cents", "neuf cents"]
        thousands = ["", "mille", "deux mille", "trois mille", "quatre mille",
                  "cinq mille", "six mille", "sept mille", "huit mille", "neuf mille"]

    if num == 0:
        return ones[0]
    words = []
    if num // 1000 > 0:
        words.append(thousands[num // 1000])
        num %= 1000
    if num // 100 > 0:
        words.append(hundreds[num // 100])
        num %= 100
    if num > 0:
        if 10 <= num < 20:
            words.append(teens[num - 10])
        else:
            if num // 10 > 0:
                words.append(tens[num // 10])
            if num % 10 > 0:
                words.append(ones[num % 10])
    return " ".join(words)

def prep(s, lang):
    new = s.split()
    numb = ['0','1','2','3','4','5','6','7','8','9']
    i = 0

    pred = ['после', 'до', 'течение', 'продолжение']
    pred_str = new[0]
    while i < len(new):
        for j in numb:
            if j in new[i]:
                nuber_str = ''
                new_str = ''
                for k in new[i]:
                    if k not in numb:
                        if nuber_str != '':
                            if pred_str in pred:
                                nuber_str = number_to_words(int(nuber_str), lang, True)
                            else:
                                nuber_str = number_to_words(int(nuber_str), lang, False)
                            new_str += ' ' + nuber_str + ' '
                        nuber_str = ''
                        new_str += k
                    else:
                        nuber_str += k
                if nuber_str != '':
                    if pred_str in pred:
                        nuber_str = number_to_words(int(nuber_str), lang, True)
                    else:
                        nuber_str = number_to_words(int(nuber_str), lang, False)
                    new_str += ' ' + nuber_str
                new[i] = new_str
        pred_str = new[i]
        if new[i] == 'би*знес' or new[i] == 'бизнес':
          new[i] = 'би*знэс'
        i += 1
    for i in range(len(new)):
        if '.' in new[i]:
            if i == len(new)-1:
                new[i] = new[i].replace('.','')
    return ' '.join(new)
  
def made_audio(s, lang, file_path = ''):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    s_new = prep0(s)
    s_new = s_new.split(".")
    ans = []
    wav = []
    if s_new[-1] == '':
        s_new.pop()
    if lang == "en" and file_path != '':
        # wav = main_tts_ssml(file_path)
        ans = wav.copy()
    else:
        for i in range(len(s_new)):
            s_prep = prep(s_new[i], lang)
            if lang == "en":
                # wav = main_tts(s_prep, "tts_models/multilingual/multi-dataset/xtts_v2", "server.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-EN.mp3", lang)
                wav = tts.tts(text=s_prep, speaker_wav="./output.wav", language=lang)
            else:
                # wav = main_tts(s_prep, "tts_models/multilingual/multi-dataset/xtts_v2", "server.wav", "/home/ubuntu/projects/kp.zuev/voicegen/TTSC/recipes/ljspeech/audio_shar/A7-RU.mp3", lang)
                wav = tts.tts(text=s_prep, speaker_wav="./output.wav", language=lang)
            ans += wav
    return ans