import argparse
import time
import inspect
import json
import pika
import requests
import os
import sys
import argparse
from rdkit.Chem import Draw, AllChem

project_path = os.path.abspath(os.path.join(os.getcwd(), "HiRXN"))
sys.path.append(project_path) #添加project路径
sys.path.append(project_path + "/HiRXN") #添加project路径下的 Hiero module路径
from HiRXN.model.main import regression
from HiRXN.log import Logger
from HiRXN.settings.settings_deployment import *


save_path = os.path.join(os.path.join(os.getcwd()),'result')
if not os.path.exists(save_path): os.mkdir(save_path)
log = Logger(os.path.join(save_path, time.strftime("%Y%m%d") + '.log'))


def save_rxn_svg(rxn_smiles, save_dir, filename):
    rxn = AllChem.ReactionFromSmarts(rxn_smiles,useSmiles=True)
    rxn_img = Draw.ReactionToImage(rxn, useSVG=True)
    file = os.path.join(save_dir, filename)
    with open(file, 'w') as f:
        f.write(rxn_img)

def runhirxn(params):
    config = argparse.Namespace(**params)
    config.radius = int(config.radius)
    if params["task_type"]=='classification':
            config.dataset='uspto_1k'
    elif params["task_type"]=='regression':
            config.dataset = 'Buchwald-Hartwig'

    if config.dataset =='Buchwald-Hartwig':
            config.max_sentence_length = 150
            config.min_count = 0
            config.class_num = 1
            config.gru_size = 50
    if config.dataset =='suzuki':
            config.max_sentence_length = 200
            config.min_count = 0
            config.class_num = 1
            config.gru_size = 50
    if config.dataset =='denmark':
            config.max_sentence_length = 100 
            config.min_count = 0
            config.class_num = 1
            config.gru_size = 50
    if config.dataset == 'uspto_1k':
            config.max_sentence_length = 400
            config.min_count = 10
            config.class_num = 1000
            config.gru_size = 500

    config.save_path = save_path
    config.cuda = True
    config.gpu = 0
    config.embedding_size = 200
    config.word2id = None
    config.seed = 0
    config.static_dir = STATIC_DIR


    log.logger.info(f"Input={json.dumps(vars(config))}")
    try:
        res = regression(config)
    except Exception as e: 
        log.logger.warning(f"There is something error! The message is {e}. Return None for Result")
        res = { 
            "task_id": f"{config.task_id}",
            "rxn_tokens": [],
            "prediction": "",
            "reaction_img_name": "non-result.png"
        }
    finally:
        log.logger.info(f"Result={res}")
    # return json.dumps(res)
    return res

def res2remote(res, post_result_url = POST_URL):
    try:
        log.logger.info(" [CONSUMER] The algorithm run successfully! Result={}".format(res))
        headers = {'Content-Type': 'application/json'}
        log.logger.info(" [CONSUMER] Send result to {}".format(post_result_url))
        response = requests.request("POST", post_result_url, headers=headers, data=json.dumps(res))
        log.logger.info(" [CONSUMER] Send Successfully! {}".format(response.text))

    except Exception as e:
        log.logger.warning("There is something error! The message is {}".format(e))

def callback_hirxn(ch, method, properties, body):
    params = json.loads(body)
    log.logger.info(f"[CONSUMER] Received Message: {params}")

    try:
        res = runhirxn(params)
        log.logger.info(f"[CONSUMER] The algorithm run successfully! Result={res}")
        res2remote(res)
        # ch.basic_ack(delivery_tag=method.delivery_tag)  # 消息响应，只有算法执行成功才回复响应
    except Exception as e:
        log.logger.warning(f"There is something error! The message is {e}")

def consumer():
    task_name = "hirxn"
    # 只要打开就会一直监听
    log.logger.info("==="*20)
    log.logger.info(f"[CONSUMER] This the for {task_name}!")
   # 1. 针对任务名字，定义 queue_name & callback_fn
    queue_name = MQ_QUEUE_NAME
    callback_fn = callback_hirxn
    
    # 2. 连接rabbitmq服务器
    credentials = pika.PlainCredentials(MQ_USERNAME, MQ_PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(MQ_HOST,MQ_PORT, 
                                  credentials=credentials,
                                  heartbeat=30*60))
    channel = connection.channel()

    # 3. 绑定交换机，并创建队列
    # channel.exchange_declare(exchange="exchange_task",  # 似乎在消费端不需要绑定交换机
    #                          exchange_type='fanout')
    channel.queue_declare(queue=queue_name,
                          durable=True,
                          # passive=True
                          )

    # 4. 执行callback
    channel.basic_qos(prefetch_count=1)  # 在同一时刻，不要发送超过一条消息给worker
    channel.basic_consume(queue=queue_name,
                          auto_ack=True, # 自动响应 只要有消息就回复响应
                          on_message_callback=callback_fn)

    # 5. 开始消费
    log.logger.info('[CONSUMER] Waiting for message. To exit press CTRL+C')
    channel.start_consuming()
    log.logger.info("==="*20 + '\n')


if __name__ == '__main__':
    consumer()
