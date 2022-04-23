import os,sys,json,re,zhconv
import torch

import logging
logger=logging.getLogger('system.tools')

def inference_ner(text,model,tokenizer,device=torch.device('cuda')):
    model_inputs=tokenizer([text],return_tensors='pt')
    input_ids=model_inputs['input_ids'].to(device)
    attention_mask=model_inputs['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_beams=4,
                            min_length=1,
                            max_length=30,
                            early_stopping=True,)#tensor([[ 102,  101, 3805, 2255, 3353, 7553,  102]], device='cuda:0')
    pred=tokenizer.decode(outputs[0],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()#'泰 山 极 顶'
    return pred


def inference_relation(text,model,tokenizer,max_length=384,device=torch.device('cuda')):
    model_inputs=tokenizer(text,return_tensors='pt',padding=True,max_length=max_length)
    input_ids=model_inputs['input_ids'].to(device)
    attention_mask=model_inputs['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_beams=4,
                            min_length=1,
                            max_length=30,
                            early_stopping=True,)
    return outputs


def select_span(tmp_q,question,ner_pred):
    start_id=tmp_q.find(ner_pred)
    end_id=start_id+len(ner_pred)
    new_pred=question[start_id:end_id]
    return new_pred

def rule_of_departure_to_destination(question,ner_pred):
    '''
    question='东营—深圳公路途径哪些城市'
    ner_pred=''东营深圳公路''
    '''
    ner_pred=''.join(ner_pred.split())
    try:
        
        if '—' in question:
            tmp_q=question.replace('—','')
            if ner_pred in tmp_q:
                start_token=ner_pred[:2]
                end_token=ner_pred[-2:]
                assert start_token in question and end_token in question
                start_pos=question.find(start_token)
                end_pos=question.find(end_token)+1
                entity=question[start_pos:end_pos+1]
                return entity
            else:
                return ''
        else:
            return ''
    
    except Exception as e:
        print(e)
        return ''

def rule_of_chinese_quotation_mark(question,ner_pred):
    '''
    question='请问“一带一路”铁路国际人才教育联盟是由哪个单位发起的？'
    ner_pred='一带一路铁路国际人才教育联盟'
    '''
    ner_pred=''.join(ner_pred.split())
    try:
        if '“' in question:
            assert '”' in question
            token_string=re.findall('“(.*)”',question)[0]
            tmp_q=question.replace('“','').replace('”','')
            if ner_pred in tmp_q:
                assert token_string in ner_pred
                new_pred=ner_pred.replace(token_string,'“{}”'.format(token_string))
                assert new_pred in question
                return new_pred
            else:
                return ''
        else:
            return ''
        
    except Exception as e:
        print(e)
        return ''
            
def rule_of_zhconv(question,ner_pred):
    '''
    question='中国共产党中央委员会政法委员会属于什么机构你知道吗？'
    ner_pred='中国共产党中央委员会政法委员會'
    '''
    ner_pred=''.join(ner_pred.split())
    if zhconv.convert(ner_pred,'zh-hans') in question:
        return zhconv.convert(ner_pred,'zh-hans')
    else:
        return ''
    
def final_rule(question,ner_pred):
    ner_pred=''.join(ner_pred.split())
    question_lower=question.lower()
    start_token=ner_pred[:2]
    end_token=ner_pred[-2:]
    
    if start_token in question_lower and end_token in question_lower:
        start_pos=question_lower.find(start_token)
        end_pos=question_lower.find(end_token)+1
        entity=question[start_pos:end_pos+1]
        return entity
    else:
        return ''

def rule1_for_find_ner(topic_entity,question,sub_map,alias_map):
    '''
    以topic_entity为起始点，往右边找
    '''
    assert topic_entity in question
    start_idx=question.find(topic_entity)
    end_idx=start_idx+len(topic_entity)
    for i in range(end_idx,len(question)):
        if question[start_idx:i] in sub_map or question[start_idx:i] in alias_map:
            topic_entity=question[start_idx:i]
            break
    
    return topic_entity

def rule2_for_find_ner(topic_entity,question,sub_map,alias_map):
    '''
    以topic_entity为终止点，往左边找
    '''
    assert topic_entity in question
    start_idx=question.find(topic_entity)
    end_idx=start_idx+len(topic_entity)
    for i in range(0,start_idx+1):
        if question[i:end_idx] in sub_map or question[i:end_idx] in alias_map:
            topic_entity=question[i:end_idx]
            break
    
    return topic_entity


def get_topic_entity(question,ner_pred):
    question_lower=question.lower()
    if ner_pred in question:
        return ner_pred
    else:     
        if ner_pred in question_lower:
            ner_pred=select_span(tmp_q=question_lower,question=question,ner_pred=ner_pred)
            
        elif rule_of_chinese_quotation_mark(question,ner_pred=ner_pred)!='':
            ner_pred=rule_of_chinese_quotation_mark(question,ner_pred=ner_pred)

        elif ''.join(ner_pred.split()) in question:
            ner_pred=select_span(tmp_q=question,question=question,ner_pred=''.join(ner_pred.split()))

        elif ''.join(ner_pred.split()) in question_lower:
            ner_pred=select_span(tmp_q=question_lower,question=question,ner_pred=''.join(ner_pred.split()))

        elif rule_of_departure_to_destination(question,ner_pred=ner_pred)!='':
            ner_pred=rule_of_departure_to_destination(question,ner_pred=ner_pred)
        
        elif rule_of_zhconv(question,ner_pred=ner_pred)!='':
            ner_pred=rule_of_zhconv(question,ner_pred=ner_pred)
            
        elif final_rule(question,ner_pred=ner_pred)!='':
            ner_pred=final_rule(question,ner_pred=ner_pred)
                
        else:
            ner_pred='未识别出实体'
        
        return ner_pred
    
    

def evaluate_ner(tmp_ner_path,alias_map,sub_map,use_rule12=True):
    if '' in alias_map:
        del alias_map['']
    if '' in sub_map:
        del sub_map['']
    ner_data=[]
    TP=0
    FP=0
    FN=0
    FP_examples=[]
    FN_examples=[]
    with open(tmp_ner_path) as f:
        lines=f.readlines()
        for line in lines:
            ner_data.append(json.loads(line.strip()))

    for example in ner_data:
        answer=example['answer']
        question=example['question']
        
        topic_entity=answer.split('|||')[0].strip()
        if ('（' in topic_entity and '）' in topic_entity):
            topic_entity=topic_entity.split('（')[0]

        try:
            assert topic_entity in question
        except:
            print(example,topic_entity)
            raise Exception("check")
        ner_pred=example['ner_pred']
        if ner_pred==topic_entity:
            TP+=1
        else:
            if ner_pred not in question:
                FN+=1#is not a true entity
                FN_examples.append(example)
                continue

            if use_rule12:
                ner_pred_rule1=rule1_for_find_ner(topic_entity=ner_pred,question=question,sub_map=sub_map,alias_map=alias_map)
                ner_pred_rule2=rule2_for_find_ner(topic_entity=ner_pred,question=question,sub_map=sub_map,alias_map=alias_map)
            else:
                ner_pred_rule1=ner_pred
                ner_pred_rule2=ner_pred
            
            if ner_pred_rule1 == topic_entity or ner_pred_rule2 == topic_entity:
                TP+=1
                if ner_pred_rule1 == topic_entity:
                    ner_pred=ner_pred_rule1
                if ner_pred_rule2 == topic_entity:
                    ner_pred=ner_pred_rule2
            else:
                if ner_pred in alias_map or ner_pred in sub_map:
                    FP+=1#is a true entity in KG but not current example
                    FP_examples.append(example)
                else:
                    FN+=1#is not a true entity. Original
                    FN_examples.append(example)

        example.update({"ner_pred":ner_pred})

    with open(tmp_ner_path,'w') as f:
        for example in ner_data:
            f.write(json.dumps(example,ensure_ascii=False)+'\n')

    logger.info("="*100)
    logger.info("TP: {}, FP: {}, FN: {}".format(TP,FP,FN))
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    f1=2*recall*precision/(recall+precision)
    logger.info("recall: {}, precision: {}, f1: {}".format(recall,precision,f1))
    logger.info("="*100)

    logger.info("Following examples are False Negative examples............")
    for e in FN_examples:
        logger.info(json.dumps(e,ensure_ascii=False))
    logger.info("Following examples are False Positive examples............")
    for e in FP_examples:
        logger.info(json.dumps(e,ensure_ascii=False))

def process_multiple_answers(all_predictions,sub_map):
    if len(all_predictions)==1:
        return all_predictions[0].split('|||')[-1].strip()
    else:
        multiple_answers=[]
        for i,pred in enumerate(all_predictions):
            head_entity=pred.split('|||')[0].strip()
            rel_nums=len(set([r for r,t in sub_map[head_entity]]))
            multiple_answers.append([pred,rel_nums])
        multiple_answers=sorted(multiple_answers,key=lambda x:x[1], reverse=True)
        results=[]
        for i,pred in enumerate(multiple_answers):
            ans=pred[0]
            results.append('答案{}：{}'.format(i+1,ans))
        return '<br>'.join(results)


def get_answers(all_alias_rel_preds,sub_map):
    all_predict_rels=set()
    for each_alias_pred in all_alias_rel_preds:
        input_q_with_prompt,predict_r=each_alias_pred
        ent_in_kg=re.findall('实体名词是：(.*)。这个句子的意图与下列哪一个关系最相似？_。',input_q_with_prompt)[0]
        
        if ''.join(predict_r.split())=='不匹配':
            continue
        
        if predict_r.lower() not in input_q_with_prompt.lower():
            predict_r=''.join(predict_r.split())
            
        if predict_r.lower() in input_q_with_prompt.lower():
            start_idx=input_q_with_prompt.lower().find(predict_r.lower())
            end_idx=start_idx+len(predict_r)
            predict_r=input_q_with_prompt[start_idx:end_idx]
            for p,o in sub_map[ent_in_kg]:
                if p==predict_r or p in predict_r or predict_r in p:
                    all_predict_rels.add(' ||| '.join([ent_in_kg,p,o]))
                    break            
            
        else:
            #Can not found predicted relation in question
            logger.info("Cannot found {} in {}".format(predict_r,input_q_with_prompt))
    return all_predict_rels