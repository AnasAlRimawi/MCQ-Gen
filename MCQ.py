from summarizer import Summarizer
from keybert import KeyBERT
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import wordnet as wn
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk
import re
import random
import streamlit as st



model = Summarizer()
kw_model = KeyBERT(model='all-mpnet-base-v2')
# kw_model = KeyBERT()
nltk.download('popular')
nltk.download('wordnet')

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):

    distractors = []
    word = word.lower()
    orig_word = word

    # replaces whitespace with underscore
    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors

    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


def get_wordsense(sent, word):
    word = word.lower()

    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def test(full_text):

    #region Text Summarizer
    result = model(full_text, min_length=60, max_length=500, ratio=0.4)
    summarized_text = ''.join(result)
    #endregion

    #region Keyword Extraction
    keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 1), stop_words=None, highlight=False,
                                         top_n=10)
    keywords_list = list(dict(keywords).keys())
    important_keyword = []
    for keyword in keywords_list:
        if keyword.lower() in summarized_text.lower():
            important_keyword.append(keyword)
    #endregion

    #region Sentence Mapping

    sentences = sent_tokenize(summarized_text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]

    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in important_keyword:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)

    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)


    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    #endregion

    #region Generate MCQ and Choices
    key_distractor_list = {}

    for keyword in keyword_sentences:
        wordsense = get_wordsense(keyword_sentences[keyword][0], keyword)
        if wordsense:
            distractors = get_distractors_wordnet(wordsense, keyword)
            #if len(distractors) == 0:
                #distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
        else:

            #distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors


    #endregion

    #region printing
    index = 1
    for each in key_distractor_list:
        sentence = keyword_sentences[each][0]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub(" _______ ", sentence)
        st.write("%s)" % (index), output)
        #print("%s)" % (index), output)
        choices = [each.capitalize()] + key_distractor_list[each]
        top4choices = choices[:4]
        #print("Correct answer is", top4choices[0])
        st.write("Correct answer is", top4choices[0])
        random.shuffle(top4choices)
        optionchoices = ['a', 'b', 'c', 'd']
        #print("\t", optionchoices[idx], ")", " ", choice)
        for idx, choice in enumerate(top4choices):
            st.write("\t", optionchoices[idx], ")", " ", choice)
        index = index + 1
    #endregion printing
    # return key_distractor_list
    # return keyword_sentences
    # return important_keyword
    # return summarized_text



