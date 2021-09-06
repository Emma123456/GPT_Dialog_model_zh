import os
import model.utils as utils
import model.dataset as dataset
import argparse
import torch
import random
import traceback
from model.model_multi_input import MultiInputModel
from model.trainer_multi_input import Trainer
from model.text import Vocab
import re
from torch.nn.parallel import DistributedDataParallel
from model.dataset import PadBatchSeq

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='/Users/cuixianyun/PycharmProjects/NLG/src/config.json')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

args = parser.parse_args()
config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'main.log'))

train_dir = os.path.join(config_path, config['train_dir'])
data_dir = os.path.join(config_path, config['data_dir'])
eval_dir = os.path.join(config_path, config['eval_dir'])
log_dir = os.path.join(config_path, config['log_dir'])
best_model = os.path.join(config_path, config['best_dir'])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

try:
    GPU_NUM = 0
    device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device("cuda", 0)
    print(device)

    vocab = Vocab(config.vocab_path)
    logger.info('Building models')
    model = MultiInputModel(config, vocab)
    model = model.to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    latest_ckpt = utils.get_latest_ckpt(train_dir)
    start_epoch = 0
    if latest_ckpt is not None:
        logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
        start_epoch = utils.get_epoch_from_ckpt(latest_ckpt)
        state_dict = torch.load(os.path.join(train_dir, latest_ckpt), map_location=device)
        model.load_state_dict(state_dict['model'], strict=True)

        test_text = ['帮我订一张到北京的火车票。询问出发地']
        test_data = []
        for text in test_text:
            post = [vocab.eos_id] + vocab.string2ids(' '.join(text)) + [vocab.eos_id]
            resp = []
            test_data.append({"post": post, "post_len": len(post), "resp": resp, "resp_len": len(resp)})

        with torch.no_grad():
            model.eval()
            samples = PadBatchSeq(model.vocab.pad_id)(test_data)
            prediction = model.predict([samples['post'].to(device)])
            for j in range(len(test_data)):
                post_str = test_text[j]
                pred_str = model.vocab.ids2string(prediction[j])
                logger.info({"post": post_str, "pred": pred_str})
    else:
        logger.error('train first')

except:
    logger.error(traceback.format_exc())
