import textblob


class TextblobDemo:
    # source_sentence = "I had a generally good experience but Megan was rude. She ignored me the first time I asked for her help and then acted like it was a huge deal. She clearly wasn't happy to be there. Other than that, everything was great. The prices were reasonable and they honored the discount that I had."
    source_sentence = "it was a wonderfull night i had in the hotel the service is v good  thanks for your hospatillity"
    sentence = textblob.Sentence(source_sentence)
    blob = textblob.TextBlob(source_sentence)
    sentences = blob.sentences
    sentences.append(sentence)

    for sub_sentence in sentences:
        print(sub_sentence)
        sub_blob = textblob.TextBlob(str(sub_sentence))
        sub_blob.correct()
        sentiment = sub_blob.sentiment
        tags = sub_blob.tags
        detected_language = sub_blob.detect_language()
        pos_tags = sub_blob.pos_tags
        tokens = sub_blob.tokens
        words = sub_blob.words
        word_counts = sub_blob.word_counts
        # noun_phrases = sub_blob.noun_phrases

        print(sentiment.polarity)
        print(sub_blob.polarity)
        print(sub_blob.subjectivity)
        print("\n")
