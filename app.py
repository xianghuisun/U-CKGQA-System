#from chatbot import chatbot
from lib2to3.pgen2 import token
from flask import Flask, request, session, redirect, url_for, render_template, flash
from inference import predict_ner, predict_relation
from bart import MyBart
from utils import read_kg, paraver
from tools import process_multiple_answers
import torch
from transformers import BertTokenizer
import numpy as np
import json
import time
import re
from argparse import ArgumentParser
import logging

#####################################################Intialize#####################################
logger=logging.getLogger('system')
logger.setLevel(logging.INFO)
fh=logging.FileHandler('log.txt',mode='w')
fh.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

device='cuda' if torch.cuda.is_available() else "cpu"
bart_model_path='/home/xhsun/NLP/huggingfaceModels/Chinese/chinese-bart-base'
checkpoint='/home/xhsun/Desktop/graduate_models/Section4/kgclue/best-model.pt'
model = MyBart.from_pretrained(bart_model_path,state_dict=torch.load(checkpoint))
model.to(device)
model.eval()
tokenizer=BertTokenizer.from_pretrained(bart_model_path)
alias_map,sub_map=read_kg(kg_path='/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt',kg='kgclue')
logger.info("alias_map and sub_map dict have been successfully loaded!")
logger.info("alias_map size is {}, sub_map size is {}".format(len(alias_map),len(sub_map)))
unanswerable_q='这个问题我暂时没学会'
app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    start_time = time.time()

    logger.info("用户问题：{}".format(query))
    ner_start_time=time.time()
    topic_entity = predict_ner(question=query,model=model,tokenizer=tokenizer)
    logger.info("该问题的主题实体是：{}".format(topic_entity))
    ner_end_time=time.time()
    logger.info("实体识别耗时：{}ms".format((ner_end_time-ner_start_time)*1000))

    match_start_time=time.time()
    all_predictions=predict_relation(question=query,entity=topic_entity,model=model,tokenizer=tokenizer,alias_map=alias_map,sub_map=sub_map)
    match_end_time=time.time()
    logger.info("匹配耗时：{}ms".format((match_end_time-match_start_time)*1000))

    end_time=time.time()
    spend_time=(end_time-start_time)*1000
    logger.info('整个QA流程总计消耗了{} ms'.format(spend_time))
    if type(all_predictions)!=list:
        all_predictions=list(all_predictions)
    if all_predictions==[]:
        logger.info(unanswerable_q)

    answer=process_multiple_answers(all_predictions=all_predictions,sub_map=sub_map)
    if answer=='' or answer=='<br>':
        return unanswerable_q
    return answer
    #return str(chatbot.get_response(userText))

if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument("--port",default=5050,help="Port")
    args=parser.parse_args()
    app.run(debug=True,host='0.0.0.0',port=args.port,use_reloader=False)
