#!/bin/env python

import sys
import json
import nltk
import random
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

with open("squad-data/dev-v1.1.json") as open_file:
    data_file = json.loads(open_file.read())
with open("verbs_by_POS.json") as open_file:
    verbs_by_POS = json.loads(open_file.read())

model = Word2Vec.load("word2vec.model")

questions = [y['question'] for d in data_file['data'] for x in d['paragraphs'] for y in x['qas']]
contexts = [x['context'] for d in data_file['data'] for x in d['paragraphs']]

def categorize_verbs(questions, contexts):
    processed_questions = [nltk.pos_tag(nltk.word_tokenize(q)) for q in questions]
    processed_contexts = [nltk.pos_tag(nltk.word_tokenize(c)) for c in contexts]
    all_verbs = []
    count = 0
    for question in processed_questions:
        verbs = list(map(lambda x : x, list(filter(lambda x : x[1] in ['VB','VBD','VBG','VBN','VBP','VBZ'], question))))
        all_verbs = all_verbs + verbs
    for context in processed_contexts:
        verbs = list(map(lambda x : x, list(filter(lambda x : x[1] in ['VB','VBD','VBG','VBN','VBP','VBZ'], context))))
        all_verbs = all_verbs + verbs

    verb_json = {}
    for i in all_verbs:  
       verb_json.setdefault(i[0],[]).append(i[1])
    for v in verb_json:
        verb_json[v] = list(set(verb_json[v]))

    POSlist = {'VB':[],'VBD':[],'VBG':[],'VBN':[],'VBP':[],'VBZ':[]}

    for v in verb_json:
            for p in verb_json[v]:
                POSlist[p].append((v, model.wv[v].tolist()))
    with open("verbs_by_POS.json", "w") as write_file:
        json.dump(POSlist, write_file)

def replace_verb(context, tagged_verb, verbs_by_POS):
    # replacements = model.wv.most_similar(verbs[-1][0], topn=20) 
    random_same_POS = verbs_by_POS[tagged_verb[1]][int(random.random()*len(verbs_by_POS[tagged_verb[1]]))][0]
    new_context = context.replace(" "+tagged_verb[0]+" "," "+random_same_POS+" ")
    return new_context

def tag_and_return_verbs(str):
    # nltk tokenize and tag a sentence
    tokens = nltk.word_tokenize(str)
    tagged = nltk.pos_tag(tokens)
    return list(map(lambda x : x, list(filter(lambda x : x[1] in ['VB','VBD','VBG','VBN','VBP','VBZ'], tagged))))

def print_modified_questions(questions, verbs_by_POS):
    #for testing
    for question in questions:
        tagged = nltk.pos_tag(nltk.word_tokenize(question))
        verbs = list(map(lambda x : x, list(filter(lambda x : x[1] in ['VB','VBD','VBG','VBN','VBP','VBZ'], tagged))))
        # arbitrarily, choose to modify the last verb
        verb_to_replace = verbs[-1]
        new_question = replace_verb(question, verb_to_replace, verbs_by_POS)
        print(question,'\n', new_question)

# create modified dev data; Take a verb from each question and modify that verb in the context accordingly.
# each modified paragraph has different context, same answers, and same question
data_paragraphs_modified = []

# for article in data_file['data'][0:10]:
#     print(article['title'])
#     new_article = {'title':article['title'], 'paragraphs':[]}
#     new_paragraphs = []
#     for paragraph in article['paragraphs']:
#         context = paragraph['context']
#         for qa in paragraph['qas']:
#             question = qa['question']
#             verbs = tag_and_return_verbs(question)
#             if not verbs == []:
#                 new_context = replace_verb(context, verbs[-1], verbs_by_POS)
#                 new_article['paragraphs'].append({'context':new_context, 'qas':[{'answers':qa['answers'],'question':qa['question'],'id':qa['id']}]})
#     data_paragraphs_modified.append(new_article)

# with open("squad-data/data_paragraphs_modified.json", "w") as write_file:
#     json.dump(data_paragraphs_modified, write_file)

# create modified dev data; Take a verb from each question and modify that verb 
# each modified paragraph has  same context, same answers, and different question
data_questions_modified = []
for article in data_file['data']:
    print(article['title'])
    new_article = {'title':article['title'], 'paragraphs':[]}
    new_paragraphs = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            verbs = tag_and_return_verbs(question)
            if not verbs == []:
                # arbitrarily, we modify the last occuring verb
                new_question = replace_verb(question, verbs[-1], verbs_by_POS)
                new_article['paragraphs'].append({'context':context, 'qas':[{'answers':qa['answers'],'question':new_question,'id':qa['id']}]})
    data_questions_modified.append(new_article)

with open("squad-data/data_questions_modified.json", "w") as write_file:
    json.dump(data_questions_modified, write_file)




