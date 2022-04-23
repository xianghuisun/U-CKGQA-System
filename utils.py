import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import random
from copy import deepcopy
import re
from flask import request
import traceback

def paraver():
    state = False
    try:
        state, conv_request = True, request.get_json(force=True)
    except Exception as error_type:
        abnormal_type = traceback.format_exc()
        state,conv_request = False,{'abnormal_type': abnormal_type}
    return state, conv_request


def read_kg(kg_path,kg):
    #'/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt'
    with open(kg_path) as f:
        lines=f.readlines()

    print('The number of triples: {}'.format(len(lines)))

    sub_map = defaultdict(set)#每一个头实体作为key，对应的所有一跳路径内的(关系，尾实体)作为value
    alias_map=defaultdict(set)
    #ent_to_relations=defaultdict(set)
    bad_line=0
    spliter='|||' if kg=='nlpcc' else '\t'
    for i in tqdm(range(len(lines))):
        line=lines[i]
        l = line.strip().split(spliter)
        s = l[0].strip()
        p = l[1].strip()
        o = l[2].strip()
        if s=='' or p=='' or o=='':
            bad_line+=1
            continue
        sub_map[s].add((p, o))

        #ent_to_relations[s].add(p)

        entity_mention=s
        if kg.lower()=='kgclue' and ('（' in s and '）' in s):
            entity_mention=s.split('（')[0]
            alias_map[entity_mention].add(s)
        if kg.lower()=='nlpcc' and ('(' in s and ')' in s):
            entity_mention=s.split('(')[0]
            alias_map[entity_mention].add(s)

        if p in ['别名','中文名','英文名','昵称','中文名称','英文名称','别称','全称','原名']:
            alias_map[entity_mention].add(o)
    return alias_map,sub_map


    
def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line))
    print("原始数据有{}个样本".format(len(data)))
    return data

# def read_data_nlpcc(data_path):
#     with open(data_path) as f:
#         lines=f.readlines()
#     i=0
#     examples=[]
#     while i+2<len(lines):
#         question=lines[i].strip().split('\t')[1]
#         i+=1
#         triple=lines[i].strip().split('\t')[1]
#         i+=1
#         try:
#             answer=lines[i].strip().split('\t')[1]
#         except:
#             print(lines[i])
#         i+=2
#         if triple.split('|||')[2].strip()==answer and len(answer)>=1:
#             examples.append({"question":question,'answer':triple})
#     print("The number examples: {} from {}".format(len(examples),data_path))
#     return examples


def read_data_nlpcc(data_path):
    with open(data_path) as f:
        lines=f.readlines()

    examples=[]
    for line in lines:
        examples.append(json.loads(line.strip()))
    
    print("The number examples: {} from {}".format(len(examples),data_path))
    return examples