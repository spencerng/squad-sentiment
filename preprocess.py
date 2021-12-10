# Data processing
# Note: it seems like the questions have IDs but the contexts don't
# Perhaps we can just hash the contents of the context as the ID?
import csv
import json
import re


with open("./squad-data/train-v1.1.json") as json_file:
  train_data = json.load(json_file)["data"]
with open("./squad-data/dev-v1.1.json") as json_file:
  dev_data = json.load(json_file)["data"]


with open('./data/contexts.csv', 'w') as csvfile:
        fieldnames = ['id', 'context']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


        writer.writeheader()
        for i in range(0, len(train_data)):
          for j in range(0, len(train_data[i]["paragraphs"])):
            writer.writerow({'id': str(i) + "x" + str(j), 'context': train_data[i]["paragraphs"][j]["context"]})


with open('./data/qa.csv', 'w') as csvfile:
        fieldnames = ['question', 'answer', 'context_id', 'start_pos', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


        writer.writeheader()
        catcnt = 0
        for i in range(0, len(train_data)):
          for j in range(0, len(train_data[i]["paragraphs"])):
            for l in range(0, len(train_data[i]["paragraphs"][j]["qas"])):
              category = 'other'
              question = train_data[i]["paragraphs"][j]["qas"][l]["question"]
              if re.search('^.*[Ww]hich\s.*\s(team)\s.*\?', question):
                category = "other entity"
              if re.search('^.*\s(day|date)\s.*\?', question):
                category = "date"
              if re.search('^.*[Ww]ho\s.*\s(player)\s.*\?', question):
                category = "person"
              if re.search('^.*[Ww]ho\s(is|was)\s.*\?', question):
                category = "person"
              if re.search('^.*[Hh]ow\s(many|long)\s.*\?', question):
                category = "other numeric"
              if re.search('^What\s(is|was)\s.*\scost\sof\s.*\?', question):
                category = "other numeric"
              if re.search('^.*\scolor\s.*\?', question):
                category = "adjective phrase"
              if re.search('^.*\slanguage\s.*\?', question):
                category = "common noun phrase"
              if re.search('^.*[Ww]here\s(did|do)\s.*\s(to|from)\s.*\?', question):
                category = "location"
              if re.search('^.*[Ww]hose\s.*\?', question):
                category = "person"
              if re.search('^.*[Ww]hat\s(year|month|day)\s.*\?', question):
                category = "date"
              if re.search('^.*[Hh]ow\sdid\s.*\?', question):
                category = "common noun phrase"
                #print(f"Q: {question}\nCAT: {category}")
              if category != 'other':
                catcnt += 1
              writer.writerow({'context_id': str(i) + "x" + str(j), 'question': train_data[i]["paragraphs"][j]["qas"][l]["question"],  'answer': train_data[i]["paragraphs"][j]["qas"][l]["answers"][0]["text"], "start_pos": train_data[i]["paragraphs"][j]["qas"][l]["answers"][0]["answer_start"], 'category': category})


with open('./data/testset.csv', 'w') as csvfile:
        fieldnames = ['question', 'answers', 'context_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        print("Categorized " + str(catcnt) + " questions")


        writer.writeheader()
        for i in range(0, len(train_data)):
          for j in range(0, len(train_data[i]["paragraphs"])):
            for l in range(0, len(train_data[i]["paragraphs"][j]["qas"])):
              a = "|".join(list(set(map(lambda x: x["text"], train_data[i]["paragraphs"][j]["qas"][l]["answers"]))))
              writer.writerow({'context_id': str(i) + "x" + str(j), 'question': train_data[i]["paragraphs"][j]["qas"][l]["question"],  'answers': a})
