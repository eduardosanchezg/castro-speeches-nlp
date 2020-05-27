import lib

tpy = lib.load_dict('tpy.json')
corpus = lib.load_dict("castro-corpus.json")
tw_xml = "politics-test-tagged.xml"

model = lib.get_doc_sentiment_prediction_model(tw_xml)

print(model.predict(["Muerte al invasor", "Viva la Revolución", "Abajo el imperialismo", "No los queremos, no los necesitamos"]))