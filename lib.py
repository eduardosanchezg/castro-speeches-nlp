# @title Lib
from bs4 import BeautifulSoup
import re
import os
import es_core_news_md
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as et
from spacy.lang.es.stop_words import STOP_WORDS
import string
from spacy.lang.es import Spanish
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
from pylab import rcParams

nlp = es_core_news_md.load()
nlp.max_length = 4000000

def import_annotated_tweets_corpus_to_df(tw_xml):
    xtree = et.parse(tw_xml)
    xroot = xtree.getroot()

    df_cols = ["Text", "Value"]
    rows = []

    for node in xroot:
        s_text = node.find("content").text
        s_value = node.find("sentiments/polarity/value").text
        if s_value == 'P':
            s_value = 1
        elif s_value == 'N':
            s_value = -1
        elif s_value == 'NEU':
            s_value = 0
        else:
            continue
        rows.append({"Text": s_text, "Value": s_value})

    return pd.DataFrame(rows, columns=df_cols)

stenographic_stops_re = [
    r'\(APLAUSOS.*?\)',
    r'\(EXCLAMACIONES.*?\)',
    r'\(ALGUIEN.*?\)',
    r'\(OVACION.*?\)',
    r'\(DEL PUBLICO.*?\)',
    r'\(RISAS.*?\)',
    r'\(.*?VERSION(.|\n)*?\)',
    r'\(.*?FIDEL CASTRO.*?\)',
    r'\(ABUCHEOS.*?\)',
    r'LOCUTOR\.\-.*?\.\n',
    r'COMDTE\. FIDEL CASTRO\.\-',
    r'REINA DE LA REFORMA AGRARIA\.\-.*?\.\n',
    r'\(SE HACE SILENCIO.*?\)',
    r'FIDEL\:',
    r'DISCURSO PRONUNCIA(.|\n)*?\.\n',
    r'NIKITA.*?\.\-.*?\.\n',
    r'\(DICEN.*?\)',
    r'CMDTE\.\-',
    r'JOSE FERIA SANCHEZ\.\-.*?\.\n',
    r'MIGUEL AMADOR\.\-.*?\.\n',
    r'JUAN ALMEIDA\.\-.*?\.\n',
    r'Discurso pronunciado por(.|\n)*?\.\n',
    r'PALABRAS DEL(.|\n)*?\.\n',
]

wrong_loc_ner = [
    "Unión Soviética",
    "URSS",
    "Comunidad Europea",
    "Viet Nam",
    "América Latina",
    "Cuito Cuanavale",
    "Sudáfrica",
    "Panamá",
    "Yugoslavia"
]

wrong_per_ner = []

def load_corpus(dir):
    corpus = {}
    for folder in os.listdir(dir):
        for file in os.listdir(dir + folder + "/esp/"):
            if not file.endswith('.html'):
                continue
            if file.startswith("mensaje"):
                continue
            key = file.strip('e.html')
            key = key.strip('a').strip('f').strip('e').strip('d').strip('c').strip('n').strip('i').strip('s')
            corpus[key] = clean_text(import_file(dir + folder + "/esp/" + file))
    return corpus

def import_file(html_file):
    return BeautifulSoup(open(html_file, encoding='windows-1252').read(), 'html.parser').get_text()

def remove_stenographic_annotations(text):
    text = str(text)
    for regex in stenographic_stops_re:
        text = re.sub(regex, '', text)
    return text

def clean_text(text):
    text = remove_stenographic_annotations(text)
    return text

def dict_to_dataframe(corpus, column_names):
    df = pd.DataFrame(list(corpus.items()), columns=column_names)
    if 'Date' in column_names:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def convert_to_yyyy_mm_dd_date(ddmmyy_date):
    d = ddmmyy_date[0:2]
    m = ddmmyy_date[2:4]
    y = ddmmyy_date[4:6]
    if int(y) < 59:
        y = "20" + y
    else:
        y = "19" + y
    return y + "-" + m + "-" + d

def speech_length_per_day(corpus):
    length_per_day = {}
    for k in corpus.keys():
        doc = nlp(corpus[k])
        length_per_day[convert_to_yyyy_mm_dd_date(k)] = len(
            [token.orth_ for token in doc if not token.is_punct | token.is_space])
    return length_per_day

def visualize_speech_length_per_day(corpus):  # todo: turn labels, group by year
    slpd_dict = speech_length_per_day(corpus)
    df = dict_to_dataframe(slpd_dict, ["Date", "Length"])
    sns.barplot(x="Date", y="Length", data=df)

    plt.show()

def speech_avg_length_by_year(corpus):
    slbd_dict = speech_length_per_day(corpus)
    year_total = {}
    year_count = {}
    for k in slbd_dict.keys():
        y = int(convert_to_yyyy_mm_dd_date(k)[0:4])
        if y not in year_total:
            year_total[y] = 0
            year_count[y] = 0
        year_total[y] += slbd_dict[k]
        year_count[y] += 1
    year_avg = dict([(year, int(year_total[year] / year_count[year])) for year in year_count.keys()])
    return year_avg

def visualize_speech_avg_length_by_year(corpus):
    slby = speech_avg_length_by_year(corpus)
    df = dict_to_dataframe(slby, ["Year", "Avg_Length"])
    sns.barplot(x="Year", y="Avg_Length", data=df)
    plt.show()

def get_yearly_corpus(corpus):
    yearly_corpus = {}
    for k in corpus.keys():
        y = int(convert_to_yyyy_mm_dd_date(k)[0:4])
        if y not in yearly_corpus:
            yearly_corpus[y] = ""
        yearly_corpus[y] += corpus[k]
    return yearly_corpus

def fix_ner(tuple):
    text, label = tuple
    if text in wrong_loc_ner:
        return (text, 'LOC')
    if text in wrong_per_ner:
        return (text, 'PER')
    else:
        return tuple

def ner_per_year(corpus):
    yearly_corpus = get_yearly_corpus(corpus)
    yearly_ner = {}
    if os.path.exists('ner_per_year.json'):
        yearly_ner = load_dict('ner_per_year.json')
    for y in yearly_corpus.keys():
        print(y)
        if str(y) in yearly_ner.keys():
            continue
        doc = nlp(yearly_corpus[y])
        yearly_ner[y] = []
        for ent in doc.ents:
            yearly_ner[y].append(fix_ner((ent.text, ent.label_)))
        save_dict(yearly_ner, 'ner_per_year.json')
    return yearly_ner

def get_toponyms_per_year(corpus):
    toponyms_per_year_dict = {}
    ner_per_year_dict = ner_per_year(corpus)
    yearly_corpus = get_yearly_corpus(corpus)
    for y in ner_per_year_dict.keys():
        toponyms_per_year_dict[y] = []
        print("---")
        print(y)
        for (ent_text, ent_label) in ner_per_year_dict[y]:
            if ent_label == 'LOC':
                toponyms_per_year_dict[y].append((ent_text, yearly_corpus[int(y)].count(ent_text)))
    return toponyms_per_year_dict

def get_people_per_year(corpus):
    people_per_year_dict = {}
    ner_per_year_dict = ner_per_year(corpus)
    yearly_corpus = get_yearly_corpus(corpus)
    for y in ner_per_year_dict.keys():
        people_per_year_dict[y] = []
        print("---")
        print(y)
        for (ent_text, ent_label) in ner_per_year_dict[y]:
            if ent_label == 'PER':
                people_per_year_dict[y].append((ent_text, yearly_corpus[int(y)].count(ent_text)))
    return people_per_year_dict

def get_freq_ent_per_year(corpus):
    freq_ent_per_year_dict = {}
    ner_per_year_dict = ner_per_year(corpus)
    yearly_corpus = get_yearly_corpus(corpus)
    for y in ner_per_year_dict.keys():
        freq_ent_per_year_dict[y] = []
        print("---")
        print(y)
        for (ent_text, ent_label) in ner_per_year_dict[y]:
            freq_ent_per_year_dict[y].append((ent_text, yearly_corpus[int(y)].count(ent_text)))
    return freq_ent_per_year_dict

def save_dict(my_dict, filename):
    json_file = json.dumps(my_dict)
    f = open(filename, "w")
    f.write(json_file)
    f.close()

def load_dict(filename):
    my_dict = {}
    with open(filename) as handle:
        my_dict = json.loads(handle.read())
    handle.close()
    return my_dict

def doc_sentiment_tokenizer(sentence):
    stopwords = list(STOP_WORDS)
    punctuations = string.punctuation
    parser = Spanish()
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    return mytokens

def get_doc_sentiment_prediction_model(tw_xml):
    if os.path.exists('sentiment_prediction_model.pkl'):
        file = open('sentiment_prediction_model.pkl', 'rb')
        model = pickle.load(file)
        file.close()
        return model
    vectorizer = CountVectorizer(tokenizer=doc_sentiment_tokenizer, ngram_range=(1, 1))
    classifier = LinearSVC()
    tw_df = import_annotated_tweets_corpus_to_df(tw_xml)
    tfvectorizer = TfidfVectorizer(tokenizer=doc_sentiment_tokenizer)
    X = tw_df['Text']
    ylabels = tw_df['Value']

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
    pipe_countvect = Pipeline([("cleaner", predictors()),
                               ('vectorizer', tfvectorizer),
                               ('classifier', classifier)])
    # Fit our data
    pipe_countvect.fit(X_train, y_train)
    file = open('sentiment_prediction_model.pkl', 'wb')
    pickle.dump(pipe_countvect, file)
    file.close()
    return pipe_countvect

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text_sent(text) for text in X]

    def fit(self, X, y, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text_sent(text):
    return text.strip().lower()

def corpus_sentiment_analysis(corpus, model):
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for k in corpus.keys():
        doc_sentiment_dict = {}
        doc = nlp(corpus[k])
        sentences = [sent.string.strip() for sent in doc.sents]
        sentiments = model.predict(sentences)
        doc_sentiment_dict[convert_to_yyyy_mm_dd_date(k)] = sum(sentiments) / len(sentiments)
    return doc_sentiment_dict

def sort_ner_dict(ner_dict):
    dict_list = list(ner_dict.items())
    dict_list = [list(x) for x in dict_list]
    for i in range(0,len(dict_list)):
        for j in range(i+1,len(dict_list)):
            if int(dict_list[i][0]) > int(dict_list[j][0]):
                temp = dict_list[i][0]
                dict_list[i][0] = dict_list[j][0]
                dict_list[j][0] = temp
    dict_list = dict(dict_list)

    for y in range(1959,2009):
        l = dict_list[str(y)]
        for i in range(0,len(l)):
            for j in range(i+1,len(l)):
                if l[i][1] < l[j][1]:
                    temp = l[i]
                    l[i] = l[j]
                    l[j] = temp
        dict_list[str(y)] = l

    return dict_list

def visualize_tpy():
    df = pd.DataFrame(columns=('year','topic','val'))
    i = 0
    rowt = dict(zip(['year', 'topic', 'val'], ['1959', 'US', 116/191]))
    row_st = pd.Series(rowt)
    row_st.name=i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1959', 'Irak', 21 / 191]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1959', 'Latin America', 19 / 191]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1959', 'China', 19 / 191]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1959', 'Geneva', 16 / 191]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1960', 'US', 85 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1960', 'Europe', 52 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1960', 'Viet Nam', 42 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1960', 'Latin America', 32 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1960', 'Chile', 20 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1961', 'Viet Nam', 250 / 799]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1961', 'US', 133 / 799]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1961', 'USSR', 174 / 799]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1961', 'Latin America', 133 / 799]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1961', 'Bulgaria', 109 / 799]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1962', 'US', 248 / 506]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1962', 'Geneva', 34 / 506]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1962', 'Mexico', 165 / 506]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1962', 'Miami', 36 / 506]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1962', 'Monterrey', 23 / 506]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1963', 'US', 149 / 404]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1963', 'Europe', 39 / 404]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1963', 'Venezuela', 89 / 404]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1963', 'Miami', 87 / 404]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1963', 'Caribbean', 40 / 404]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1964', 'US', 609 / 998]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1964', 'Europe', 131 / 998]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1964', 'Latin America', 82 / 998]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1964', 'China', 95 / 998]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1964', 'Caribbean', 81 / 998]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1964', 'Caribbean', 81 / 998]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1965', 'Viet Nam', 126 / 306]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1965', 'USSR', 43 / 306]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1965', 'US', 64 / 306]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1965', 'Argelia', 37 / 306]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1965', 'Latin America', 36 / 306]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1966', 'Ethiopia', 73 / 195]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1966', 'US', 36 / 195]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1966', 'Africa', 43 / 195]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1966', 'Somalia', 23 / 195]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1966', 'Latin America', 20 / 195]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1967', 'Angola', 141 / 509]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1967', 'Africa', 130 / 509]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1967', 'US', 98 / 509]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1967', 'Jamaica', 77 / 509]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1967', 'Mozambique', 63 / 509]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1968', 'Angola', 57 / 193]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1968', 'Latin America', 45 / 193]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1968', 'US', 42 / 193]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1968', 'Cuito Cuanavale', 27 / 193]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1968', 'Africa', 22 / 193]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1969', 'Nicaragua', 84 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1969', 'Mexico', 67 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1969', 'Latin America', 41 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1969', 'Venezuela', 39 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1969', 'US', 67 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1970', 'US', 97 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1970', 'Grenada', 83 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1970', 'Nicaragua', 29 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1970', 'Central America', 15 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1970', 'Africa', 15 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1971', 'US', 156 / 426]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1971', 'Viet Nam', 90 / 426]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1971', 'Santo Domingo', 85 / 426]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1971', 'Latin America', 69 / 426]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1971', 'Argelia', 26 / 426]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1972', 'US', 84 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1972', 'Nicaragua', 60 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1972', 'Central America', 40 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1972', 'Angola', 28 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1972', 'El Salvador', 27 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1973', 'Latin America', 237 / 611]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1973', 'US', 208 / 611]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1973', 'Brazil', 72 / 611]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1973', 'Mexico', 48 / 611]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1973', 'Caribbean', 46 / 611]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1974', 'US', 147 / 268]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1974', 'Nicaragua', 34 / 268]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1974', 'USSR', 44 / 268]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1974', 'Latin America', 26 / 268]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1974', 'Africa', 17 / 268]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1975', 'US', 152 / 397]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1975', 'Latin America', 95 / 397]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1975', 'USSR', 114 / 397]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1975', 'Venezuela', 20 / 397]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1975', 'Guatemala', 16 / 397]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1976', 'US', 249 / 406]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1976', 'Latin America', 48 / 406]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1976', 'Miami', 41 / 406]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1976', 'Iraq', 39 / 406]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1976', 'Viet Nam', 29 / 406]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1977', 'Latin America', 97 / 565]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1977', 'US', 241 / 565]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1977', 'Venezuela', 138 / 565]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1977', 'Santo Domingo', 61 / 565]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1977', 'Miami', 28 / 565]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1978', 'US', 41 / 190]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1978', 'Polonia', 36 / 190]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1978', 'Angola', 35 / 190]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1978', 'Africa', 35 / 190]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1978', 'USSR', 43 / 190]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1979', 'US', 146 / 478]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1979', 'USSR', 174 / 478]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1979', 'Latin America', 76 / 478]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1979', 'Europe', 46 / 478]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1979', 'Spain', 36 / 478]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1980', 'US', 79 / 152]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1980', 'Latin America', 30 / 152]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1980', 'Nicaragua', 23 / 152]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1980', 'Europe', 12 / 152]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1980', 'Mexico', 8 / 152]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1981', 'US', 165 / 375]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1981', 'Latin America', 63 / 375]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1981', 'Venezuela', 42 / 375]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1981', 'Mexico', 38 / 375]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1981', 'USSR', 67 / 375]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1982', 'US', 34 / 99]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1982', 'Latin America', 31 / 99]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1982', 'Europe', 15 / 99]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1982', 'Rome', 10 / 99]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1982', 'USSR', 9 / 99]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1983', 'US', 66 / 154]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1983', 'Latin America', 38 / 154]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1983', 'USSR', 22 / 154]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1983', 'Europe', 16 / 154]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1983', 'Miami', 12 / 154]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1984', 'US', 144 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1984', 'Brazil', 37 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1984', 'Iraq', 21 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1984', 'USSR', 17 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1984', 'Viet Nam', 12 / 231]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1985', 'Chile', 158 / 421]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1985', 'Latin America', 99 / 421]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1985', 'US', 106 / 421]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1985', 'Viet Nam', 30 / 421]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1985', 'Europe', 28 / 421]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1986', 'Viet Nam', 163 / 371]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1986', 'Chile', 76 / 371]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1986', 'US', 67 / 371]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1986', 'Czechoslovakia', 38 / 371]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1986', 'Latin America', 27 / 371]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1987', 'USSR', 264 / 599]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1987', 'US', 138 / 599]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1987', 'Latin America', 120 / 599]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1987', 'Europe', 56 / 599]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1987', 'Brazil', 21 / 599]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1988', 'Angola', 158 / 379]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1988', 'US', 76 / 379]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1988', 'Panama', 64 / 379]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1988', 'Africa', 40 / 379]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1988', 'USSR', 41 / 379]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1989', 'US', 130 / 248]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1989', 'USSR', 72 / 248]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1989', 'Grenada', 17 / 248]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1989', 'Nicaragua', 16 / 248]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1989', 'Viet Nam', 13 / 248]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1990', 'Venezuela', 150 / 474]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1990', 'US', 114 / 474]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1990', 'Latin America', 99 / 474]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1990', 'Viet Nam', 79 / 474]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1990', 'Caracas', 32 / 474]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1991', 'US', 51 / 104]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1991', 'Caribbean', 13 / 104]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1991', 'USSR', 16 / 104]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1991', 'China', 12 / 104]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1991', 'Europe', 12 / 104]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1992', 'US', 82 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1992', 'Latin America', 59 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1992', 'USSR', 79 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1992', 'Namibia', 50 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1992', 'Panama', 28 / 298]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1993', 'US', 105 / 316]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1993', 'Latin America', 88 / 316]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1993', 'USSR', 75 / 316]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1993', 'Europe', 25 / 316]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1993', 'Brazil', 23 / 316]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1994', 'US', 133 / 304]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1994', 'USSR', 85 / 304]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1994', 'Latin America', 34 / 304]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1994', 'Europe', 28 / 304]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1994', 'Angola', 24 / 304]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1995', 'Viet Nam', 153 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1995', 'Latin America', 78 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1995', 'Chile', 75 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1995', 'Puerto Rico', 30 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1995', 'China', 28 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1996', 'US', 255 / 569]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1996', 'Latin America', 125 / 569]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1996', 'Costa Rica', 58 / 569]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1996', 'Washington', 57 / 569]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1996', 'USSR', 74 / 569]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1997', 'US', 520 / 933]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1997', 'Latin America', 116 / 933]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1997', 'Mexico', 105 / 933]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1997', 'Caribbean', 96 / 933]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1997', 'Africa', 96 / 933]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1998', 'US', 112 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1998', 'Latin America', 55 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1998', 'USSR', 32 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1998', 'Mexico', 20 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1998', 'Guatemala', 20 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1999', 'US', 242 / 352]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1999', 'Latin America', 38 / 352]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1999', 'Europe', 29 / 352]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1999', 'Argentina', 25 / 352]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['1999', 'Barbados', 18 / 352]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2000', 'US', 85 / 269]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2000', 'Viet Nam', 77 / 269]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2000', 'Nicaragua', 50 / 269]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2000', 'Latin America', 29 / 269]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2000', 'China', 28 / 269]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2001', 'US', 112 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2001', 'Viet Nam', 35 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2001', 'USSR', 33 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2001', 'Europe', 32 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2001', 'China', 27 / 239]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2002', 'US', 98 / 266]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2002', 'Angola', 66 / 266]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2002', 'Africa', 36 / 266]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2002', 'Korea', 35 / 266]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2002', 'South Africa', 31 / 266]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2003', 'US', 90 / 192]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2003', 'Venezuela', 54 / 192]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2003', 'Miami', 19 / 192]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2003', 'Bolivia', 18 / 192]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2003', 'Caribbean', 11 / 192]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2004', 'US', 71 / 137]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2004', 'Viet Nam', 13 / 137]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2004', 'USSR', 19 / 137]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2004', 'Latin America', 18 / 137]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2004', 'Florida', 16 / 137]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2005', 'US', 224 / 428]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2005', 'Europe', 59 / 428]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2005', 'Venezuela', 51 / 428]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2005', 'Washington', 51 / 428]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2005', 'Brazil', 43 / 428]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2006', 'US', 217 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2006', 'Venezuela', 54 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2006', 'Miami', 49 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2006', 'Latin America', 23 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2006', 'Africa', 21 / 364]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2007', 'US', 277 / 631]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2007', 'Latin America', 184 / 631]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2007', 'USSR', 63 / 631]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2007', 'Brazil', 58 / 631]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2007', 'Miami', 49 / 631]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2008', 'Viet Nam', 139 / 369]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2008', 'US', 110 / 369]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2008', 'Holstein', 47 / 369]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2008', 'Latin America', 43 / 369]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    rowt = dict(zip(['year', 'topic', 'val'], ['2008', 'Peru', 30 / 369]))
    row_st = pd.Series(rowt)
    row_st.name = i
    i = i + 1
    df = df.append(row_st)

    df['year'] = pd.to_datetime(df['year'])

    #topic_year_notnormal = df.pivot(index='topic', columns='year', values='val')
    #topic_year_notnormal.fillna(0, inplace=True)
    #t_y_nn = topic_year_notnormal.transpose()
    topic_year = df.pivot_table(index='topic', columns='year', values='val')
    topic_year.fillna(0, inplace=True)

    t_y = topic_year.transpose()

    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
                      '#800080', '#FF00FF', '#000080', '#0000FF', '#008080',
                      '#00FFFF', '#008000', '#00FF00', '#808000', '#FFFF00',
                      '#800000', '#FF0000', '#000000', '#808080', '#C0C0C0',
                      '#FFFFFF', '#B22222', '#F08080', '#FF69B4', '#FF7F50',
                      '#FFFF00',
                      '#BDB76B', '#BC8F8F', '#8B4513', '#00008B', '#FFE4E1',
                      '#008080',
                      '#00CED1', '#00CED1']

    rcParams['font.family'] = 'Verdana'
    rcParams['font.size'] = 16
    rcParams['legend.fontsize'] = 8
    rcParams['figure.figsize'] = 25, 10
    rcParams['legend.frameon'] = True
    # labels = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014']
    plt.axis([0, 6, 0, 20])
    # plt.xticks(labels)
    t_y.plot.area(alpha=0.5)

    plt.show()

def visualize_doc_sentiment():
    doc_sentiment_dict = load_dict('doc_sentiment_dict.json')
    df = dict_to_dataframe(doc_sentiment_dict,['Date','Polarity'])
    sns.lineplot(x="Date", y="Polarity", data=df)
    plt.show()