#!/usr/bin/python3

def message_rating(message):
    import joblib
    from sklearn.feature_extraction.text import CountVectorizer

    classifier = joblib.load('model.joblib')
    vector = joblib.load('vectorizer.pkl')

    test = vector.transform([message])

    final = classifier.predict(test)
    chances = round((float(classifier.predict_proba(test)[0][1] * 100)), 5)

    if final == 'spam':
        return_statement = 'This message is spam' + '\n' + 'This message has a probability of {0}% of being spam'.format(chances)
    else:
        return_statement = 'This message is not spam' + '\n' + 'This message has a probability of {0}% of being spam'.format(chances)

    return str(return_statement)


