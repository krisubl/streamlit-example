from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import os
import subprocess
import io 
from sklearn.feature_extraction.text import CountVectorizer
from io import BytesIO
import base64
from datetime import datetime, date
import numpy as np
import pytz
import re
import nltk
import spacy
import string
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ast
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import itertools
from googletrans import Translator
from spacy.lang.id import Indonesian
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer

#nltk.download('stopwords')
#nltk.download('punkt')

# # Install Node.js (because tweet-harvest built using Node.js)
# !sudo apt-get update
# !sudo apt-get install -y ca-certificates curl gnupg
# !sudo mkdir -p /etc/apt/keyrings
# !curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

# !NODE_MAJOR=20 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list

# !sudo apt-get update
# !sudo apt-get install nodejs -y

# !node -v

# os.system("sudo apt-get update")
# os.system("sudo apt-get install -y ca-certificates curl gnupg")
# os.system("sudo mkdir -p /etc/apt/keyrings")
# os.system("curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg")
# NODE_MAJOR = 20
# os.system(f'echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_{NODE_MAJOR}.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list')
# os.system("sudo apt-get update")
# os.system("sudo apt-get install nodejs -y")
# os.system("node -v")

#VARIABEL GLOBAL
os.environ['filename']= 'bps.csv'
file_analisis=[]
data_bers=[]
file_crawling=[]
action_triggered_visual = False
action_run= False
lexicon_file = 'sentiwords_id.txt'


def score(x):
    blob1 = TextBlob(x)
    return blob1.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Fungsi untuk membersihkan satu tweet
def clean_tweet(tweet):
    # Hapus @username
    cleaned_tweet = re.sub(r'@\w+', '', tweet)

    # Hapus spasi ekstra
    cleaned_tweet = ' '.join(cleaned_tweet.split())

    cleaned_tweet = re.sub(r'@\w+', '', tweet)

    # Hapus URL
    cleaned_tweet = re.sub(r'http\S+', '', cleaned_tweet)

    # Hapus tanda baca
    cleaned_tweet = cleaned_tweet.translate(str.maketrans('', '', string.punctuation))

    return cleaned_tweet

def clean_text_lib(text):
    return cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)

# def clean_tweets(tweet):
#     url1="https://raw.githubusercontent.com/evanmartua34/Twitter-COVID19-Indonesia-Sentiment-Analysis---Lexicon-Based/master/cleaning_source/update_combined_slang_words.txt"

#     try:
#         # Mengambil konten dari URL pertama
#         response1 = requests.get(url1)
#         response1.raise_for_status()
#         content1 = response1.text
#         print("sukses")
#     except requests.exceptions.RequestException as e:
#         print("Gagal mengambil data:", str(e))
 
#     slang_words = ast.literal_eval(content1)

#     # my_file.close()
#     # file_2.close()

#     tweet = tweet.lower()
#     #after tweepy preprocessing the colon left remain after removing mentions
#     #or RT sign in the beginning of the tweet
#     tweet = re.sub(r':', '', tweet)
#     tweet = re.sub(r'‚Ä¶', '', tweet)
#     #replace consecutive non-ASCII characters with a space
#     tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

#     #remove emojis from tweet
#     #tweet = emoji_pattern.sub(r'', tweet)

#     #remove punctuation manually
#     tweet = re.sub('[^a-zA-Z]', ' ', tweet)

#     #remove tags
#     tweet=re.sub("&lt;/?.*?&gt;","&lt;&gt;",tweet)

#     #remove digits and special chars
#     tweet=re.sub("(\\d|\\W)+"," ",tweet)

#     #remove other symbol from tweet
#     tweet = re.sub(r'â', '', tweet)
#     tweet = re.sub(r'€', '', tweet)
#     tweet = re.sub(r'¦', '', tweet)

#     word_tokens = word_tokenize(tweet)
#     for w in word_tokens:
#         if w in slang_words.keys():
#             word_tokens[word_tokens.index(w)] = slang_words[w]

def count_words(x):
    words = word_tokenize(x)
    n=len(words)
    return n

def del_word(x,key_list):
    n = len(key_list)
    word_tokens = word_tokenize(x)
    new_x =''
    for word in word_tokens:
        if word not in key_list:
            new_x = new_x+word+' '
    return new_x

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='id', dest='en')
    return translated.text

kamus_normalisasi_3={"@": "di", "abis": "habis", "ad": "ada", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik", "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur", "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana", "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur", "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci", "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor", "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan", "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile", "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi", "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda", "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo", "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh", "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi", "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif ", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus", "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah", "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba", "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras", "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual", "tioghoa": "tionghoa", "ngentotin": "senggama", "kentot": "senggama", "faktakta": "fakta", "sohib": "teman", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin"}

#1. STOPWORD ELIMINATION
def tokenisasi(text):
  tokens = text.split(" ")
  return tokens

nlp = Indonesian() # use directly
nlp = spacy.blank('id') # blank instance'
stopwords = nlp.Defaults.stop_words

def stopword_elim(text):
  tokens = tokenisasi(text)
  tokens_nostopword = [w for w in tokens if not w in stopwords]
  return (" ").join(tokens_nostopword)

#2. STEMMING
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
  # stemming process
  output = stemmer.stem(text)
  return output

def stemming_sentence(text):
  output = ""
  for token in tokenisasi(text):
    output = output + stemming(token) + " "
  return output[:-1]

kamus_normalisasi_2=pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')
#kamus_normalisasi_2
kamus_normalisasi_test=dict(zip(kamus_normalisasi_2['slang'],kamus_normalisasi_2['formal']))
kamus_normalisasi_test.update(kamus_normalisasi_3)

def normalisasi(string):
  output = ""
  for t in string.split(" "):
    output = output + (kamus_normalisasi_test[t] if t in kamus_normalisasi_test else t) + " "
  return output

#FINAL PREPROCESSING
def finalpreprocess(string):
 return stemming_sentence(stopword_elim(normalisasi(string)))

vs = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    sentiment = vs.polarity_scores(text)
    return sentiment

def vader_analysis(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <=  -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def sentimen_label(sentiment_score):
  if sentiment_score > 0:
    sentiment = "Positive"
  elif sentiment_score < 0:
      sentiment = "Negative"
  else:
      sentiment = "Neutral"
  return sentiment

def analyze_sentiment(text, lexicon_data):
    words = text.split()
    sentiment_score = 0

    for word in words:
        if word in lexicon_data:
            sentiment_score += lexicon_data[word]
    
    label = sentimen_label(sentiment_score)
    return label


#PREPROCESSING
def Cleaning_Data(df):
    df['clean_text'] = df['tweets'].copy()
    # df = df.drop_duplicates()
    # df = df.reset_index(drop=True)
    df['clean_text'] = df['clean_text'].apply(clean_tweet)
    # #df['clean_text'] = df['clean_text'].apply(clean_tweets)
    # #df['text'] = df['clean_text'].apply(lambda x: del_word(x,keyword))
    # #df['english_text'] = df['clean_text'].apply(translate_to_english)
    df['clean_text'] = df['clean_text'].apply(lambda x: finalpreprocess(x))
    
    # df['Score'] = df['text'].apply(lambda x: vs.polarity_scores(x))
    # df['Compound Score'] = df['text'].apply(lambda x: vs.polarity_scores(x)['compound'])


    #st.write(df.head(10))
    return df



@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.title('ANALISIS SENTIMEN')

st.header('Crawling Data')
with st.expander('Enter Kata Kunci'):
    kata_kunci=st.text_input('Masukkan Kata Kunci')
    t_dari=st.date_input("Masukkan Tanggal Mulai", pd.Timestamp("2023-01-01"))
    t_sampai=st.date_input("Masukkan Tanggal Selesai")
    # in_dari= datetime.strptime(t_dari, "%Y/%m/%d")
    # in_sampai= datetime.strptime(t_sampai, "%Y/%m/%d")
    e_since = t_dari.strftime("%Y-%m-%d")
    e_until = t_sampai.strftime("%Y-%m-%d")
    #os.environ['filename']
    lim=st.text_input('Masukkan Limit')
    os.environ['limit']=lim
    os.environ['search_keyword'] = f"{kata_kunci} since:{e_since} until:{e_until}"
    st.write("$search_keyword")

    if st.button("RUN", key="runnin") :
        #st.write(search_keyword)
        #my_api_token = '0d94f81cb49e38b4692746400a9013d835e56613'
        os.environ['MY_API_TOKEN'] ='0d94f81cb49e38b4692746400a9013d835e56613'
        command = f'npx tweet-harvest@latest -o "$filename" -s "$search_keyword" -l 500 --token "$MY_API_TOKEN"'

        try:
            subprocess.run(command, shell=True, check=True)
            action_run= True
            file_path = f"tweets-data/bps.csv"
            # Read the CSV file into a pandas DataFrame
            dfb = pd.read_csv(file_path, delimiter=";")
            #dfb = pd.read_csv(file_path)
            st.write(dfb)  # Menampilkan DataFrame
            file_crawling=dfb
            crawl = convert_df(dfb)
            # st.download_button(
            #     label="Download data as CSV",
            #     data=crawl,
            #     file_name='hasil_crawling.csv',
            #     mime='text/csv',
            # )
        except subprocess.CalledProcessError as e:
            print(f"Perintah gagal dijalankan: {e}")

        

with st.expander('Download Hasil Crawling'):
    if action_run:
        crawl = convert_df(file_crawling)
        st.download_button(
            label="Download data as CSV",
            data=crawl,
            file_name='hasil_crawling.csv',
            mime='text/csv',
            )
    
# with st.expander('MASIH PROSES ~ Olah Data Hasil Crawling'):
#     if st.button("Preprocessing Data", key="preproscessing"):
#         #action_triggered_visual = True 
#         #File hasil crawling didefinisikan disini
#         #file_analisis=df
#         st.write("Menampilkan tabel hasil preprocessing")
#         clean_data = convert_df(filename)
#         st.download_button(
#             label="Download data as CSV",
#             data=clean_data,
#             file_name='data_hasil_preprocessing.csv',
#             mime='text/csv',
#             )
        
#     if st.button("MASIH PROSES ~ Menganalisis Sentimen", key="analisis"):
#         #file hasil crawling
#         # df_crawl="file.csv"
#         # df_crawl['score'] = df_crawl['full_text'].apply(score)
#         # df_crawl['sentiment'] = df_crawl['score'].apply(analyze)
#         # st.write(df_crawl.head(10))

#         # hasil_sentimen = convert_df(df_crawl)

#         st.download_button(
#             label="Download data as CSV",
#             data=hasil_sentimen,
#             file_name='sentiment_crawling.csv',
#             mime='text/csv',
#         )
#     if st.button("TEKAN SKRG BIKIN ERROR-Analisis Data Hasil Preprocessing", key="aksi2"):
#         action_triggered_visual = True 
#         #File hasil crawling didefinisikan disini
#         #file_analisis=df
#         #Jika ditekan sekarang akan munculin error di bawah soalnya belum buat yg crawl untuk didefinisikan
#         st.write("BELUM JADI~ Scroll ke Bawah untuk Melihat Hasil Visualisasi")


st.header('Sentiment Analysis by Input')
with st.expander('Analisis Teks'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))


    pre = st.text_input('Clean Text: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analisis CSV'):
    upl = st.file_uploader('Upload file')

#     def score(x):
#         blob1 = TextBlob(x)
#         return blob1.sentiment.polarity

# #
#     def analyze(x):
#         if x >= 0.5:
#             return 'Positive'
#         elif x <= -0.5:
#             return 'Negative'
#         else:
#             return 'Neutral'


    if upl:
        df = pd.read_excel(upl)
        # del df['Unnamed: 0']
        #2 ROW DIBAWAH di COMMENT DULU YA
        # df['score'] = df['tweets'].apply(score)
        # df['sentiment'] = df['score'].apply(analyze)
        st.write(df.head(10))

        # @st.cache_data
        # def convert_df(df):
        #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
        #     return df.to_csv().encode('utf-8')

        #Mendefinisikan file yang akan divisualisasikan
        #file_analisis=df

        #COmment dulu
        # csv = convert_df(df)

        # st.download_button(
        #     label="Download data as CSV",
        #     data=csv,
        #     file_name='sentiment.csv',
        #     mime='text/csv',
        #)
    
    #text_key = st.text_input('Masukkan Kata kunci (agar tidak menjadi dominasi keyword pada visualisasi): ')
    

    if st.button("Analisis Data", key="clean_dari_file"):
        
    #     teks="coba terjemahkan kata"
    #     trs=translate_to_english(teks)
    #     st.write(trs)
        # df['tweets'] = cleantext.clean(df['tweets'], clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
        df['tweets']= df['tweets'].apply(clean_text_lib)
        data_bers=Cleaning_Data(df)
        
        #data_bersih=pd.DataFrame(data_bersih)
        # data_bers['score'] = data_bers['clean_text'].apply(score)
        # data_bers['sentiment'] = data_bers['score'].apply(analyze)
        
        st.write(data_bers.head(10))
        
    #     st.write("Cleanng Data Telah Selesai ~ Masih Dalam Tahap Pengembangan")

    # if st.button("Analisis Sentimen", key="sen_analisis"):
    
    #     #st.write(data_bersih.head(10))
    #     #data_bersih['score'] = data_bersih['clean'].apply(score)
    #     #data_bersih['sentiment'] = data_bersih['score'].apply(analyze)
    #     #st.write(data_bersih.head(10))
    #     st.write("Cleanng Data Telah Selesai ~ Masih Dalam Tahap Pengembangan")


    #ANALISIS 2
        lexicon_data = {}
        with open(lexicon_file, 'r') as file:
            for line in file:
                word, score = line.strip().split(':')  # Misalkan lexicon tersimpan dalam format "kata skor"
                lexicon_data[word] = int(score)  # Konversi skor ke tipe data integer

        # Menganalisis sentimen dan menambahkan kolom baru ke DataFrame
        data_bers['sentiment'] = data_bers['tweets'].apply(lambda text: analyze_sentiment(text, lexicon_data))

        # Simpan DataFrame dengan kolom sentimen ke file XLSX
        #data_bers.to_excel('file_output_sentimen.xlsx', index=False)
            #if st.button("Gunakan File", key="aksi"):
                #st.write(data_bers.head(10))

        st.write("data bers SENTIMEN")
        st.write(data_bers.head(10))
        action_triggered_visual = True 
        #st.write(file_analisis.head(10))
        st.write("FILE ANALISIS")
        file_analisis=data_bers
        st.write(file_analisis.head(10))
        csv = convert_df(file_analisis)
        # with st.spinner('Sedang memproses...'):
        #     st.download_button(
        #         label="Download data as CSV",
        #         data=file,
        #         file_name='sentiment.csv',
        #         mime='text/csv',
        #     )
        st.write("data_bers")



st.header('Visualisasi')
with st.expander('Perbandingan Sentimen'):
    st.write("Menampilkan diagram batang sentimen negatif, positif, netral")
    if action_triggered_visual:
        # # Menampilkan grafik batang
        st.title("Grafik Sentimen")
        sentiment_counts = file_analisis['sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', color='pink', figsize=(6, 4), ax=ax)
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)
        # file_analisis['sentiment'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))
    #if st.button("Download Diagram", key="batang"):
        # Template pengecekan aja
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)

with st.expander('Persentase Sentimen'):
    st.write("Menampilkan Pie Chart sentimen negatif, positif, netral")
    if action_triggered_visual:
        st.title("Persentase Sentimen")
        sentiment_counts = file_analisis['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_counts.plot.pie(colors=['pink', 'lightblue', 'lightgreen'], autopct='%1.2f%%', startangle=90, ax=ax)
        ax.set_ylabel('')  
        st.pyplot(fig)
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)

with st.expander('Menampilkan 10 Sentimen Netral'):
    st.write("Menampilkan Tabel berisi 10 tweet teratas dengan sentimen netral")
    if action_triggered_visual:
        st.write(file_analisis[file_analisis['sentiment'] == "Neutral"].head(10))
        tab_netral = convert_df(file_analisis['sentiment'] == "Neutral")
        st.download_button(
            label="Download data as CSV",
            data=tab_netral,
            file_name='top_sentiment_netral.csv',
            mime='text/csv',
            )
    
with st.expander('Menampilkan 10 Sentimen Positif'):
    st.write("Menampilkan Tabel berisi 10 tweet teratas dengan sentimen positif")
    if action_triggered_visual:
        st.write(file_analisis[file_analisis['sentiment'] == "Positive"].head(10))
        tab_positif = convert_df(file_analisis['sentiment'] == "Positive")
        st.download_button(
            label="Download data as CSV",
            data=tab_positif,
            file_name='top_sentiment_positif.csv',
            mime='text/csv',
            )

with st.expander('Menampilkan 10 Sentimen Negatif'):
    st.write("Menampilkan Tabel berisi 10 tweet teratas dengan sentimen negatif")
    if action_triggered_visual:
        st.write(file_analisis[file_analisis['sentiment'] == "Negative"].head(10))
        tab_negatif = convert_df(file_analisis['sentiment'] == "Negative")
        st.download_button(
            label="Download data as CSV",
            data=tab_negatif,
            file_name='top_sentiment_negatif.csv',
            mime='text/csv',
            )

with st.expander('Menampilkan 30 Kata yang Sering Muncul'):
    st.write("Menampilkan Bar Chart yang berisikan 30 kata yang sering Muncul")
    if action_triggered_visual:
        # Daftar kata-kata stop dalam bahasa Indonesia
        stop_words_id = ["dan", "atau", "dengan", "yang", "pada", "ke", "dari", "di", "ini", "itu", "saya", "kamu", "dia", "kita", "mereka"]

        # Inisialisasi CountVectorizer dengan daftar kata-kata stop bahasa Indonesia
        cv = CountVectorizer(stop_words=stop_words_id)
        words = cv.fit_transform(file_analisis.clean_text)

        sum_words = words.sum(axis=0)

        words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)


        frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

        # Membangun aplikasi Streamlit
        frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
        plt.title("Most Frequently Occuring Words - Top 30")
        st.bar_chart(frequency.head(30).set_index('word'))
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)

with st.expander('Menampilkan WordCloud Semua Tweet'):
    st.write("Menampilkan WordCloud")
    if action_triggered_visual:
        wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))
        # Menampilkan tabel frekuensi kata-kata
        st.image(wordcloud.to_array(), caption="WordCloud Semua Tweet", use_column_width=True)
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)

with st.expander('Menampilkan WordCloud Tweet Sentimen Netral'):
    st.write("Menampilkan WordCloud Netral")
    if action_triggered_visual:
        normal_words =' '.join([text for text in file_analisis['clean_text'][file_analisis['sentiment'] == 'Neutral']])
        wordcloud1 = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
        st.image(wordcloud1.to_array(), caption="WordCloud Bersentimen Netral", use_column_width=True)
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)

with st.expander('Menampilkan WordCloud Tweet Sentimen Positif'):
    st.write("Menampilkan WordCloud Positif")
    if action_triggered_visual:
        pos_words =' '.join([text for text in file_analisis['clean_text'][file_analisis['sentiment'] == 'Positive']])
        wordcloud2 = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(pos_words)
        st.image(wordcloud2.to_array(), caption="WordCloud Bersentimen Positif", use_column_width=True)
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)

with st.expander('Menampilkan WordCloud Tweet Sentimen Negatif'):
    st.write("Menampilkan WordCloud Negatif")
    if action_triggered_visual:
        neg_words =' '.join([text for text in file_analisis['clean_text'][file_analisis['sentiment'] == 'Negative']])
        wordcloud3 = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(neg_words)
        st.image(wordcloud3.to_array(), caption="WordCloud Bersentimen Negatif", use_column_width=True)
        # Menyimpan plot sebagai gambar PNG
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        # Membuat tautan unduhan untuk gambar
        href = f'<a href="data:image/png;base64,{base64.b64encode(img_stream.read()).decode()}" download="contoh_plot.png">Unduh Gambar</a>'
        st.markdown(href, unsafe_allow_html=True)
