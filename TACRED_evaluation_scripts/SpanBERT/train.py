import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dataloader import DataLoader
from trainer import BERTtrainer
from utils import torch_utils, scorer, constant, helper

from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.set_defaults(lower=False)

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=10, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--device', type=int, default=0, help='gpu device to use.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')

args = parser.parse_args()

opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

tokenizer = BertTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')
special_tokens_dict = {'additional_special_tokens': constant.ENTITY_TOKENS}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, tokenizer, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, tokenizer,  evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

# model
if not opt['load']:
    trainer = BERTtrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file'] 
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = BERTtrainer(model_opt)
    trainer.load(model_file)  

trainer.encoder.model.resize_token_embeddings(len(tokenizer)) 

id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
current_lr = opt['lr']

global_step = 0
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch, epoch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    x = 0
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for _, batch in enumerate(dev_batch):
        preds = trainer.predict(batch, id2label, tokenizer)
        predictions += preds
    predictions = [id2label[p] for p in predictions]
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
        train_loss, dev_loss, dev_f1))
    dev_score = dev_f1
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or dev_score > max(dev_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_score*100))
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1]:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))
