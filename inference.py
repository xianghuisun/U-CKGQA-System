import random
from tools import inference_ner, inference_relation
from tools import get_topic_entity, get_answers

import logging
logger=logging.getLogger("system.inference")

def predict_ner(question,model,tokenizer):
    text=question.lower().replace('“','').replace('”','').replace('"','').replace('"','').replace('\n','').replace('\t','')
    text='{}'.format(text)+'。上个句子中哪些属于实体名词？_。'#'你知道泰山极顶的作家是谁啊？。上个句子中哪些属于实体名词？_。'
    ner_pred=inference_ner(text,model=model,tokenizer=tokenizer)
    ner_pred=get_topic_entity(question=question,ner_pred=ner_pred)
    logger.info("Input question: {}, predicted named entity: {}".format(question,ner_pred))
    return ner_pred


def predict_relation(question,entity,model,tokenizer,alias_map,sub_map):
    alias_map[entity].add(entity)
    pred_results=[]
    model_inputs=[]
    logger.info("{} has {} alias entities in KG".format(entity,len(alias_map[entity])))
    logger.info("All alias entities are list following: {}".format('  '.join(list(alias_map[entity]))))
    for alias_ent in alias_map[entity]:
        #找到这个主题实体对应的所有别名实体
        relations=[]
        for r,t in sub_map[alias_ent]:
            relations.append(r)
        #获取这个别名实体的所有关系
        
        candidate_relations=relations
        candidate_relations.append('不匹配')
        random.shuffle(candidate_relations)#确保“不匹配”这个选项不总是在最后一个位置
        
        relation_string=[]
        for i,rel in enumerate(candidate_relations):
            relation_string.append('（{}）'.format(i)+rel)
            
        relation_string='，'.join(relation_string)
        text='{}'.format(question)+'。'+'实体名词是：{}'.format(alias_ent)+"。这个句子的意图与下列哪一个关系最相似？_。"+' '+relation_string
        model_inputs.append(text)
    
    try:
        assert model_inputs!=[]
    except Exception as e:
        logger.info("question: {}, entity: {}, model inputs is null list".format(question,entity))
        return []

    outputs=inference_relation(model_inputs,model=model,tokenizer=tokenizer)
    #assert outputs.size(0)==len(model_inputs)
    for input_,output in zip(model_inputs,outputs):
        pred=tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
        pred_results.append([input_,pred])
        logger.info("input: {}, output: {}".format(input_,pred))
    #pred_results means rel_pred

    all_predictions=get_answers(all_alias_rel_preds=pred_results,sub_map=sub_map)
    logger.info("Predicted answers are list following:")
    for pred in all_predictions:
        logger.info(pred)

    return all_predictions
    