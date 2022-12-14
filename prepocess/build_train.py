import os
import json
import pickle
import argparse
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm

random.seed(7)
catgeory_number = 8
subcate_number = 4
subcate_news = 4


def find_index(clist, cv):
    for i in range(len(clist)):
        if clist[i] == cv:
            return i
    return -1


def build_examples(rank, args, df, news_info, user_info, fout, uemb):
    data_list = []
    for imp_id, uid, imp in tqdm(df[["id", "uid", "imp"]].values, total=df.shape[0]):
        if uid not in user_info:
            continue

        cur_emb = uemb[user_info[uid]['idx']]
        cur_category = {cur_emb[cindex]: cindex for cindex in range(catgeory_number)}
        cur_subcate = {cur_emb[subcindex]: subcindex - catgeory_number for subcindex in
                       range(catgeory_number, catgeory_number + catgeory_number * subcate_number)}

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

        neg_num = len(imp_neg_list)
        if neg_num < args.neg_count:
            imp_neg_list += ['<pad>'] * (args.neg_count - neg_num)

        cef_dict = {}
        cef_dict['<pad>'] = {
            'cate_index': 0,
            'lt': 0,
            'subcate_index': 0,
            'ls': 0
        }
        for nstr in imp_pos_list + imp_neg_list:
            if nstr in cef_dict:
                continue

            cef_dict[nstr] = {}
            nstr_cate = news_info[nstr]['category']
            nstr_subcate = news_info[nstr]['subcate']
            # 若当前样本新闻属于用户历史点击的主题，则lt置为1且cate_index置为该主题index对应索引（0-7），否则lt置为0且cate_index置为0；
            if nstr_cate in cur_category:
                cef_dict[nstr]['cate_index'] = cur_category[nstr_cate]
                cef_dict[nstr]['lt'] = 1
            else:
                cef_dict[nstr]['cate_index'] = 0
                cef_dict[nstr]['lt'] = 0

            if nstr_subcate in cur_subcate:
                cef_dict[nstr]['subcate_index'] = cur_subcate[nstr_subcate]
                cef_dict[nstr]['ls'] = 1
            else:
                cef_dict[nstr]['subcate_index'] = 0
                cef_dict[nstr]['ls'] = 0

        for p in imp_pos_list:
            sampled = random.sample(imp_neg_list, args.neg_count)
            new_row = []
            new_row.append(int(imp_id))
            new_row.append(0)
            # idx
            new_row.append(news_info[p]['idx'])
            for neg in sampled:
                new_row.append(news_info[neg]['idx'])
            # cate
            new_row.append(cef_dict[p]['cate_index'])
            for neg in sampled:
                new_row.append(cef_dict[neg]['cate_index'])
            # subcate
            new_row.append(cef_dict[p]['subcate_index'])
            for neg in sampled:
                new_row.append(cef_dict[neg]['subcate_index'])
            # coefficient t
            new_row.append(cef_dict[p]['lt'])
            for neg in sampled:
                new_row.append(cef_dict[neg]['lt'])
            # coefficient s
            new_row.append(cef_dict[p]['ls'])
            for neg in sampled:
                new_row.append(cef_dict[neg]['ls'])

            new_row.append(user_info[uid]['idx'])

            data_list.append(new_row)

    datanp = np.array(data_list, dtype=int)
    np.save(fout, datanp)
    print(datanp.shape)


def main(args):
    #  Path of the training samples file (data/train/behaviors.tsv)
    f_train_beh = os.path.join(args.root, args.fsamples)
    df = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    news_info = json.load(open('{}/news.json'.format(args.root), 'r', encoding='utf-8'))
    user_info = json.load(open('{}/user.json'.format(args.root), 'r', encoding='utf-8'))
    user_embd = np.load('{}/user_emb.npy'.format(args.root))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        # Path of the output dir (data/raw/train-{i}.npy)
        output_path = os.path.join(args.root, args.fout, "train-{}.npy".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, user_info, output_path, user_embd))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    data_list = []
    for i in range(args.processes):
        file_name = os.path.join(args.root, args.fout, "train-{}.npy".format(i))
        data_list.append(np.load(file_name))
        os.remove(file_name)
    datanp = np.concatenate(data_list, axis=0)

    sub_len = math.ceil(len(datanp) / args.file_num)

    for i in range(args.file_num):
        s = i * sub_len
        e = (i + 1) * sub_len
        np.save(os.path.join(args.root, args.fout, "train-{}-new.npy".format(i)), datanp[s: e])

    print(datanp.shape, sub_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="train/behaviors.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--neg_count", default=4, type=int)
    parser.add_argument("--processes", default=10, type=int, help="Processes number")
    parser.add_argument("--file_num", default=4, type=int, help="final train file number")
    parser.add_argument("--root", default="data", type=str)
    args = parser.parse_args()

    main(args)
