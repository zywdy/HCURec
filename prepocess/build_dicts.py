import pandas as pd
from tqdm import tqdm
import os
import re
import json
import numpy as np


def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return [k["WikidataId"] for k in json.loads(x)]


punctuation = '!,;:?"\''


def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip().lower()


def parse_his(user_dict, his_beh, desc, user_idx):
    for uid, hist in tqdm(his_beh[['uid', 'hist']].values, total=his_beh.shape[0], desc=desc):

        if uid not in user_dict:
            user_dict[uid] = {"hist": [], "idx": user_idx}
            user_idx += 1
        else:
            if len(user_dict[uid]['hist']) > 0:
                continue

        hist_list = str(hist).split(' ')
        if str(hist) == 'nan':
            continue

        user_dict[uid]["hist"] = hist_list

    return user_idx


data_path = 'data'
max_title_len = 30
max_entity_len = 5

print("Loading news info")
f_train_news = os.path.join(data_path, "train/news.tsv")
f_dev_news = os.path.join(data_path, "dev/news.tsv")
# f_test_news = os.path.join(data_path, "test/news.tsv")

train_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                         names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                         quoting=3)
dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                       names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                       quoting=3)
# test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
#                         names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
#                         quoting=3)

# all_news = pd.concat([train_news, dev_news, test_news], ignore_index=True)
all_news = pd.concat([train_news, dev_news], ignore_index=True)
all_news = all_news.drop_duplicates("newsid")

news_dict = {}
word_dict = {'<pad>': 0}
entity_dict = {'<pad>': 0}
category_dict = {'<pad>': 0, '<his>': 1}
subcategory_dict = {'<pad>': 0, '<his>': 1}

word_idx = 1
news_idx = 2
entity_idx = 1
category_idx = 2
subcategory_idx = 2

for n, title, cate, subcate, title_ents, abs_ents in tqdm(
        all_news[['newsid', "title", "cate", "subcate", "title_ents", "abs_ents"]].values, total=all_news.shape[0],
        desc='parse news'):
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(title).split()  # 去掉标题中的标点并按空格分割
    wid_arr = []
    for t in tarr:
        if t not in word_dict:  # 若不在词典中，则word_idx+1
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < max_title_len:
        for l in range(max_title_len - cur_len):
            wid_arr.append(word_dict['<pad>'])
    news_dict[n]['title'] = wid_arr[:max_title_len]

    entity_arr = parse_ent_list(title_ents) + parse_ent_list(abs_ents)
    eidx_arr = set()
    for e in entity_arr:
        if e not in entity_dict:
            entity_dict[e] = entity_idx
            entity_idx += 1
        eidx_arr.add(entity_dict[e])
    eidx_n = list(eidx_arr)
    ent_len = len(eidx_n)
    if ent_len < max_entity_len:
        for l in range(max_entity_len - ent_len):
            eidx_n.append(entity_dict['<pad>'])
    news_dict[n]['entity'] = eidx_n[:max_entity_len]

    if cate not in category_dict:
        category_dict[cate] = category_idx
        category_idx += 1
    news_dict[n]['category'] = category_dict[cate]

    if subcate not in subcategory_dict:
        subcategory_dict[subcate] = subcategory_idx
        subcategory_idx += 1
    news_dict[n]['subcate'] = subcategory_dict[subcate]

# paddning news for impression
news_dict['<pad>'] = {}
news_dict['<pad>']['idx'] = 0
tarr = removePunctuation("This is the title of the padding news").split()
wid_arr = []
for t in tarr:
    if t not in word_dict:
        word_dict[t] = word_idx
        word_idx += 1
    wid_arr.append(word_dict[t])
cur_len = len(wid_arr)
if cur_len < max_title_len:
    for l in range(max_title_len - cur_len):
        wid_arr.append(word_dict['<pad>'])
news_dict['<pad>']['title'] = wid_arr[:max_title_len]
news_dict['<pad>']['entity'] = [entity_dict['<pad>']] * max_entity_len
news_dict['<pad>']['category'] = category_dict['<pad>']
news_dict['<pad>']['subcate'] = subcategory_dict['<pad>']

## paddning news for history
news_dict['<his>'] = {}
news_dict['<his>']['idx'] = 1
news_dict['<his>']['title'] = [word_dict['<pad>']] * max_title_len
news_dict['<his>']['entity'] = [entity_dict['<pad>']] * max_entity_len
news_dict['<his>']['category'] = category_dict['<his>']
news_dict['<his>']['subcate'] = subcategory_dict['<his>']

print('news_num', len(news_dict))  # 125592
print('entity num', len(entity_dict))  # 42192
print('category num', len(category_dict))  # 20
print('subcategory num', len(subcategory_dict))  # 281
print('word num', len(word_dict))  # 52181

user_dict = {}
user_idx = 0

f_his_beh = os.path.join(data_path, "train/behaviors.tsv")
his_beh1 = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
user_idx = parse_his(user_dict, his_beh1, 'parse train behaviors', user_idx)

f_his_beh = os.path.join(data_path, "dev/behaviors.tsv")
his_beh2 = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
user_idx = parse_his(user_dict, his_beh2, 'parse dev behaviors', user_idx)

f_his_beh = os.path.join(data_path, "test/behaviors.tsv")
his_beh3 = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
user_idx = parse_his(user_dict, his_beh3, 'parse test behaviors', user_idx)


imp_list2 = []
imp_pos = []
imp_dict = {}
for n, uid, time, his, imp in tqdm(
        his_beh3[["id", "uid", "time", "hist", "imp"]].values, total=his_beh3.shape[0],
        desc='parse test_beh'):
    imp_list = str(imp).split(' ')
    imp_pos_list = []
    imp_neg_list = []
    for impre in imp_list:
        arr = impre.split('-')
        curn = arr[0]
        label = int(arr[1])

        if label == 0:
            imp_neg_list.append(curn)
        elif label == 1:
            imp_pos_list.append(curn)
        else:
            raise Exception('label error!')
    for nstr in imp_pos_list + imp_neg_list:
        imp_list2.append(news_dict[nstr]['idx'])
    for p in imp_pos_list:
        imp_pos.append(news_dict[p]['idx'])

for key in imp_list2:
    imp_dict[key] = imp_dict.get(key, 0) + 1
imp_pos_dict = {}
for key in imp_pos:
    imp_pos_dict[key] = imp_pos_dict.get(key, 0) + 1
for i in imp_dict:
    if i in imp_pos_dict:
        imp_dict[i] = imp_pos_dict[i] / imp_dict[i]
    else:
        imp_dict[i] = 0

for nid, ninfo in tqdm(news_dict.items(), total=len(news_dict), desc='parse news dict'):
    if ninfo['idx'] in imp_dict:
        news_dict[nid]['pop'] = imp_dict[ninfo['idx']]
    else:
        news_dict[nid]['pop'] = 0


cate_num_list = []
subcate_num_list = []
cate_news_list = []
subcate_news_list = []
# item()方法把字典中两个每对key和value组成一个元组，并把这些元组放在列表中返回
# "cate_dict": {"11": {"101": [6894, 15557]}, "5": {"46": [10051], "64": [26359]}...}
# {"category":{"subcate": [属于该主题和子主题的新闻idx]}...}
for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='parse user dict'):
    cate_dict = {}
    for h in uinfo['hist']:
        cate = news_dict[h]['category']
        subcate = news_dict[h]['subcate']
        idx = news_dict[h]['idx']
        if cate not in cate_dict:
            cate_dict[cate] = {}
        if subcate not in cate_dict[cate]:
            cate_dict[cate][subcate] = []
        cate_dict[cate][subcate].append(idx)

    uinfo['cate_dict'] = cate_dict

    cate_num_list.append(len(cate_dict))
    for ck, cd in cate_dict.items():
        total_num = 0
        subcate_num_list.append(len(cd))
        for sbk, sbl in cd.items():
            subcate_news_list.append(len(sbl))
            total_num += len(sbl)
        cate_news_list.append(total_num)

avg_cate_num = sum(cate_num_list) / len(cate_num_list)
avg_subcate_num = sum(subcate_num_list) / len(subcate_num_list)
avg_cate_news = sum(cate_news_list) / len(cate_news_list)
avg_subcate_news = sum(subcate_news_list) / len(subcate_news_list)
print('avg category per user', avg_cate_num)  # 6.84
print('avg subcategory per category', avg_subcate_num)  # 1.95
print('avg news number per category', avg_cate_news)  # 3.35
print('avg news number per subcategory', avg_subcate_news)  # 1.72

print('save dicts')
json.dump(user_dict, open(os.path.join(data_path, "user.json"), 'w', encoding='utf-8'))
json.dump(news_dict, open(os.path.join(data_path, "news.json"), 'w', encoding='utf-8'))
json.dump(word_dict, open(os.path.join(data_path, "word.json"), 'w', encoding='utf-8'))
json.dump(entity_dict, open(os.path.join(data_path, "entity.json"), 'w', encoding='utf-8'))
json.dump(category_dict, open(os.path.join(data_path, "category.json"), 'w', encoding='utf-8'))
json.dump(subcategory_dict, open(os.path.join(data_path, "subcategory.json"), 'w', encoding='utf-8'))
