import os
import sys
import pdb
import argparse
import time
from datetime import datetime
from pathlib import Path
import yaml
import pprint
import random
import numpy as np
import platform
from torch.utils.data.sampler import SequentialSampler
from datasets.data_loader import return_dataset
from models import create_model
from trainer import Trainer
from utils.common_utils import (
    getLogger, set_seed, save_checkpoint_best_only, trim, 
    collect_results_for_analysis)

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
 
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                              help="choose from { train | test }")
    train_parser.add_argument('--cfg', type=str, default="configs/volleyball.yml",
                              help="config file path")
    train_parser.add_argument("--model_resume", type=int, required=False,
                              help="whether to load pretrained weights for training")
    train_parser.add_argument("--checkpoint", type=str, required=False,
                              help="a path to model checkpoint file to load pretrained weights")
    train_parser.add_argument('--not_save_best_model', action='store_true', 
                              help="not save best model in this run")
    
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                             help="choose from { train | test }")
    test_parser.add_argument('--cfg', type=str, default="configs/volleyball.yml",
                             help="config file path")
    test_parser.add_argument("--checkpoint", type=str, required=False,
                             help="a path to model checkpoint file to load pretrained weights") 

    args = parser.parse_args()
    
    
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    args = parser.parse_args()

    args.dev = 'cuda:' + str(args.dev)
    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1

    
    # if args.checkpoint_dir:
    #     Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        
    return args


def training_main(args):
    
    start_time = time.time()
    
    curr_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    logfile_path = os.path.join(args.log_dir, args.exp_name + '-' + curr_time + 
                                    '-' +  args.mode + '.log')
    
    logger = getLogger(name=__name__, path=logfile_path)
    
    if args.seed > 0:
        set_seed(args.seed)
    else:
        args.seed = random.randint(0, 1000000)
        set_seed(args.seed)
    
    logger.info("Working config: {}\n".format(args))
    logger.info("Host: {}".format(platform.node()))
    logger.info("Logfile will be saved in {}".format(logfile_path))
    

    train_dataset = return_dataset(args.dataset_name, args, train_model=True)
    logger.info('total number of clips is {} for training data'.format(train_dataset.__len__()))

    test_dataset = return_dataset(args.dataset_name, args, train_model=False)
    logger.info('total number of clips is {} for testing data'.format(test_dataset.__len__()))

    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers, 
                              pin_memory=True)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers, 
                             pin_memory=True)
        
    model = create_model(args, logger)
    

    criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.group_activities_weights)
    criterion_person = torch.nn.CrossEntropyLoss(weight=train_dataset.person_actions_weights)

            
            
    if args.model_resume:  
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        params = checkpoint['state_dict']
        params = trim(params)
        model.module.load_state_dict(
            params, strict=False) if hasattr(model, 'module') else model.load_state_dict(params, strict=False)
        
        logger.info("Loaded checkpoint from {}".format(args.checkpoint))

    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    model=nn.DataParallel(model, device_ids=args.gpu).cuda()

    #model = model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    
    
    trainer = Trainer(args, model, logger, criterion, criterion_person, optimizer,
                      train_loader=train_loader, test_loader=test_loader)
    
     
    best_prec1 = 0
    best_prec3 = 0
    best_i=0
    best_c=0
    best_gi=0
    best_T=0
    best_max=0
    best_max_id=0
    best_epoch = -1
    prec_dict={
        0:"prec1",
        1:"prec1i",
        2:"prec1c",
        3:"prec1gi",
        4:"preTotal1"
    }
    for epoch in range(1, args.num_epochs + 1):
        
             
        if args.gpu:
            torch.cuda.empty_cache()
        scheduler.step()
        trainer.train(epoch)
        logger.info(f"Finished Training epoch-{epoch}")
        
        prec1, prec3,  loss,prec1i,prec1c,prec1gi,preTotal1= trainer.test(epoch)
        
        logger.info(f"Finished Testing epoch-{epoch}")
        curr_prec=[prec1,prec1i,prec1c,prec1gi,preTotal1]
        curr_max=max(curr_prec)
        curr_max_id=curr_prec.index(curr_max)
        # remember best and save checkpoint
        is_best = curr_max > best_max
        best_prec1=max(best_prec1,prec1)
        best_i=max(best_i,prec1i)
        best_c = max(best_c, prec1c)
        best_gi = max(best_gi, prec1gi)
        best_T = max(best_T, preTotal1)

        if is_best:
            best_prec3 = prec3
            best_epoch = epoch
            best_max=curr_max
            best_max_id=curr_max_id

            # save the model if it is the best so far
            if not args.not_save_best_model:
                save_checkpoint_best_only(
                    {'cfg': args,
                     'epoch': epoch,
                     'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                     'top1': prec1,
                     'top3': prec3,
                     'best_top1': best_prec1,
                     'best_top3': best_prec3,
                     'optimizer': optimizer.state_dict()
                    },  
                    dir=args.checkpoint_dir, 
                    name=args.exp_name + '-' + curr_time+'@'+str(epoch)+'@'+str(prec1))
                 

        
        logger.info(f"Test Prec@1: {prec1:.3f} % / Prec@3: {prec3:.3f} %    @epoch-{epoch}")
        logger.info(
            f"== prec1i: {prec1i:.3f} % / prec1c: {prec1c:.3f} % / prec1gi: {prec1gi:.3f} % / preTotal1: {preTotal1:.3f} %    @epoch-{epoch}")

        logger.info(f"**Best Test: {best_max:.3f} % / best_max_id: {prec_dict[best_max_id]}   @epoch-{best_epoch}")
        logger.info(f"Best Test Prec@1: {best_prec1:.3f} % / Prec@3: {best_prec3:.3f} %    @epoch-{best_epoch}")
        logger.info(f"Best Test best_i: {best_i:.3f} % / best_c: {best_c:.3f} % / best_gi: {best_gi:.3f} % / best_t: {best_T:.3f}   @epoch-{best_epoch}")

        logger.info(f"Checkpoint of epoch-{best_epoch} is/was saved to {args.checkpoint_dir}.")
        
         
         
    logger.info("Done training in {} seconds.".format(time.time() - start_time))

    return




def testing_main(args):
    start_time = time.time()
    
    exp_logname = args.checkpoint.split('/')[-1].split('_model_best.pth')[0]
    
    logfile_path = os.path.join(args.log_dir, exp_logname + '-' +  args.mode + '.log')
    
    logger = getLogger(name=__name__, path=logfile_path)

    if args.seed > 0:
        set_seed(args.seed)
    else:
        args.seed = random.randint(0, 1000000)
        set_seed(args.seed)
        
    logger.info("Working config: {}\n".format(args))
    logger.info("Host: {}".format(platform.node()))
    print("Logfile will be saved in {}".format(logfile_path))
    
    
    test_dataset = return_dataset(args.dataset_name, args, train_model=False)
    logger.info('total number of clips is {} for testing data'.format(test_dataset.__len__()))
    
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers, 
                              pin_memory=True)
     
    
    model = create_model(args, logger)
    
    
    if args.use_group_activity_weights:
        criterion = torch.nn.CrossEntropyLoss(weight=test_dataset.group_activities_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion.to(args.dev)
 
    if args.use_person_action_weights:
        criterion_person = torch.nn.CrossEntropyLoss(weight=test_dataset.person_actions_weights)
    else:
        criterion_person = torch.nn.CrossEntropyLoss()
    criterion_person.to(args.dev)
    
    
    optimizer = None
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    params = checkpoint['state_dict']
    params = trim(params)
    model.module.load_state_dict(
        params, strict=False) if hasattr(
        model, 'module') else model.load_state_dict(params, strict=False)
        
    logger.info("Loaded checkpoint from {}".format(args.checkpoint))

    # Model Num_Params
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    
    
    # Trainer
    trainer = Trainer(args, model, logger, criterion, criterion_person, optimizer, 
                      test_loader=test_loader)
    
    # Going to start testing
    epoch = checkpoint['epoch']
        
    if args.gpu:
        torch.cuda.empty_cache()

          

    prec1, prec3, loss, prec1i, prec1c, prec1gi, preTotal1 = trainer.test(epoch)
    logger.info(f"Finished Testing epoch-{epoch}")
    logger.info(f"=== Test Result ===  Prec@1: {prec1:.3f} % / Prec@3: {prec3:.3f} %      @epoch-{epoch}")
    logger.info(f"== prec1i: {prec1i:.3f} % / prec1c: {prec1c:.3f} % / prec1gi: {prec1gi:.3f} % / preTotal1: {preTotal1:.3f} %    @epoch-{epoch}")

    
    
    logger.info("Done testing in {} seconds.".format(time.time() - start_time))
    
    return


if __name__ == '__main__':
    
    args = get_args_parser()
    if args.mode == "train":
        training_main(args)
    elif args.mode == "test":
        testing_main(args)
    else:
        print("Wrong mode!")
        os._exit(0)
