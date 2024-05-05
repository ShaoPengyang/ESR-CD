import json
import random
from collections import defaultdict
import pdb
import pandas as pd
import numpy as np

min_log = 0


def divide_data():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    with open('../data/ASSIST/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    l_log = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        else:
            l_log += stus[stu_i]['log_num']
        stu_i += 1
    # 2. divide dataset into train_set and test_set
    traineval_set, test_set = [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * 0.8)
        test_size = stu['log_num'] - train_size
        logs = []
        for log in stu['logs']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[train_size:]
        # shuffle logs in train_slice together, get train_set
        # knowledge id list
        for log in stu_train['logs']:
            traineval_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
        for log in stu_test['logs']:
            test_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
    random.shuffle(traineval_set)
    cut = int(0.875*len(traineval_set))
    train_set = traineval_set[:cut]
    eval_set = traineval_set[cut:]
    print("training length whole: " + str(len(train_set)))
    print("eval_set length whole: " + str(len(eval_set)))
    print("testing length whole: " + str(len(test_set)))

    with open('../data/ASSIST/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/ASSIST/eval_set.json', 'w', encoding='utf8') as output_file:
        json.dump(eval_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/ASSIST/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set


def divide_data_weak_coverage():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    with open('../data/ASSIST/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    l_log = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        else:
            l_log += stus[stu_i]['log_num']
        stu_i += 1
    # 2. read all data, obtain dict = [knowledge_id:times] for each user
    # and record which concepts are for training and testing for each user.
    train_c, test_c = defaultdict(list), defaultdict(list)
    for stu in stus:
        dict1 = defaultdict(int)
        user_id = stu['user_id']
        for log in stu['logs']:
            exer_id = log['exer_id']
            score = log['score']
            # list
            knowledge_codes = log['knowledge_code']
            for concept_id in knowledge_codes:
                dict1[concept_id] += 1

        all_keys = list(dict1.keys())
        num_concepts = len(all_keys)
        cut = int(num_concepts * 0.2)
        
        test_c[user_id].extend(all_keys[:cut])
        train_c[user_id].extend(all_keys[cut:])

    traineval_set, test_set = [], []
    for stu in stus:
        user_id = stu['user_id']
        for log in stu['logs']:
            if len(set(test_c[user_id]).intersection(set(log['knowledge_code']))) > 0:
                test_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                                  'knowledge_code': log['knowledge_code']})
            else:
                traineval_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                                 'knowledge_code': log['knowledge_code']})

    cut2 = int(0.875*len(traineval_set))
    train_set = traineval_set[:cut2]
    eval_set = traineval_set[cut2:]
    print("training length whole: " + str(len(train_set)))
    print("eval_set length whole: " + str(len(eval_set)))
    print("testing length whole: " + str(len(test_set)))
    pdb.set_trace()

    with open('../data/ASSIST/train_set_ood.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/ASSIST/eval_set_ood.json', 'w', encoding='utf8') as output_file:
        json.dump(eval_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/ASSIST/test_set_ood.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set



if __name__ == '__main__':
    # divide_data()
    divide_data_weak_coverage()
