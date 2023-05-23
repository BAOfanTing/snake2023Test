import os
import time
import argparse
import datetime
import numpy as np
from collections import defaultdict
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor,load_pretained
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from collections import Counter

# # 定义图像增强方式
# tta_transform = transforms.Compose([
#     # transforms.ToPILImage(),
#     # transforms.RandomCrop((384, 384)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     # transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
#
#
# # 定义 TTA 策略
# def tta(image, meta, model):
#     # 使用数据增强方法对输入数据进行多次变换
#     outputs = []
#     num_classes = None
#     for i in range(5):
#         # 对图像进行增强
#         image = tta_transform(image)
#         # 使用模型进行预测
#         _, output = model(image,meta)
#         if num_classes is None:
#             num_classes = output.shape[1]
#         output = output.argmax(dim=1).cpu().numpy().tolist()
#         # print(output)
#         outputs.append(output)
#     outputs = np.array(outputs)
#     final_output = [Counter(outputs[:,j].tolist()).most_common(1)[0][0] for j in range(outputs.shape[1])]  # label index
#     ### convert to onehot format
#     final_output = torch.tensor(final_output, device=image.device).unsqueeze(dim=1)
#     onehot = torch.zeros(image.shape[0], num_classes, dtype=image.dtype, device=image.device)
#     onehot = onehot.scatter(1,final_output,1)
#     return onehot



def parse_option():
    parser = argparse.ArgumentParser('MetaFG training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path',default='/root/autodl-tmp/data', type=str, help='path to dataset')
    parser.add_argument('--pretrained_cfg',default=None)
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    parser.add_argument('--num-workers', type=int, 
                        help="num of workers on dataloader ")
    
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        help='weight decay (default: 0.05 for adamw)')
    
    parser.add_argument('--min-lr', type=float,
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float,
                        help='warmup learning rate')
    parser.add_argument('--epochs', type=int,
                        help="epochs")
    parser.add_argument('--warmup-epochs', type=int,
                        help="epochs")
    
    parser.add_argument('--dataset', type=str,
                        help='dataset')
    parser.add_argument('--lr-scheduler-name', type=str,
                        help='lr scheduler name,cosin linear,step')
    
    parser.add_argument('--pretrain', type=str,
                        help='pretrain')
    
    parser.add_argument('--tensorboard', action='store_true', help='using tensorboard')
    
    
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    

    if config.FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Linear(model.head.in_features, model.head.out_features)
        print('*' * 20 + ' freeze backbone ' + '*' * 20)

    model.cuda()
    logger.info(str(model))  # TODO 临时屏蔽

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    if config.MODEL.PRETRAINED:
        load_pretained(config,model_without_ddp,logger)
        if config.EVAL_MODE:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            return

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        logger.info(f"**********normal test***********")
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        #acc1, acc5, loss = validate(config, data_loader_val, model,epoch)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            logger.info(f"**********normal test***********")
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


@torch.no_grad()
def validate(config, data_loader, model,mask_meta=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    snake_test_result = []

    end = time.time()
    for idx, data in enumerate(tqdm(data_loader)):
        if config.DATA.ADD_META:
            images,target,meta = data
            #images = images[:,4,...].view((-1, images.shape[2], images.shape[3], images.shape[4]))
            meta = [m.float() for m in meta]
            meta = torch.stack(meta,dim=0)
            if mask_meta:
                meta = torch.zeros_like(meta)
            meta = meta.cuda(non_blocking=True)
            
        else:
            images, target = data
            meta = None
        
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if config.DATA.ADD_META:
            #tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(384,384))
            #output = tta(images,meta,model)
            output = model(images,meta)
        else:
            output = model(images)
######  
        #output = output
        _,output = output
######
        if config.DATA.DATASET in ['snakeclef2023test', 'snakeclef2023valid']:
            for idx_b in range(len(target)):
                snake_test_result.append((target[idx_b].cpu(), output[idx_b].cpu()))
            continue
##############
        #loss = criterion(output, target)
        _,loss = criterion(output, target)
#################
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    if config.DATA.DATASET == 'snakeclef2023test':
        if config.DATA.ADD_META and mask_meta:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2023test_mask_meta.tc')
        else:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2023test.tc')
        torch.save(snake_test_result, output_file)
        print(len(snake_test_result), output_file)
    elif config.DATA.DATASET == 'snakeclef2023valid':
        if config.DATA.ADD_META and mask_meta:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2023valid_mask_meta.tc')
        else:
            output_file = os.path.join(config.OUTPUT, 'result_snakeclef2023valid.tc')
        torch.save(snake_test_result, output_file)
        print(len(snake_test_result), output_file)

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

def load_csv_annotations(fp, row_length=6):
    lines = []
    with open(fp, 'r') as rf:
        cr = csv.reader(rf)
        for row in cr:
            if len(row) < row_length:
                continue
            if row[0] == 'observation_id':
                names = row
                continue
            lines.append({name : row[idx] for idx, name in enumerate(names)})
    return lines


def get_class_and_idx_mapper():
    root = '../data'
    items = load_csv_annotations(fp=os.path.join(root, 'train_split.csv'))
    classes = set()
    for it in items:
        classes.add(it['class_id'])
    classes = sorted(list(classes))
    class_to_idx = {name:ind for ind,name in enumerate(classes)}
    return classes, class_to_idx


def get_items():
    root = '../data'
    items = load_csv_annotations(fp=os.path.join(root, 'SnakeCLEF2023-Test.csv'), row_length=5)
    return items



if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
###
    #torch.cuda.set_device(config.LOCAL_RANK)\
###
    torch.cuda.set_device(0)
    #If your operating system is Windows, 'backend'use gloo
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}",local_rank=config.LOCAL_RANK)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    main(config)
    model_result_file = './output/MetaFG_meta_2/OUTPUT_TAG/result_snakeclef2023test.tc'
    snake_test_result = torch.load(model_result_file)
    item_list = get_items()
    classes, class_to_idx = get_class_and_idx_mapper()
    print(len(classes), len(class_to_idx))


    observation_to_classes = defaultdict(list)
    print(len(snake_test_result))
    for idx_st, st in enumerate(snake_test_result):
        v1, v2 = st
        data_item = item_list[int(v1.detach().cpu().numpy())]
        sm_scores = torch.nn.Softmax(dim=0)(v2).detach().cpu().numpy()
        idx_max = np.argmax(sm_scores)
        #print(classes)
        #print(idx_max)
        cls_max = classes[idx_max]
        observation_to_classes[data_item['observation_id']].append((cls_max, sm_scores[idx_max]))
    print(len(observation_to_classes))


    result_list = []
    for obv_id, class_id_scores in observation_to_classes.items():
        sorted_scores = sorted(class_id_scores, key=lambda x: x[1], reverse=True)
        select_cls_id_according_to_max_score = sorted_scores[0][0]
        # Get most.
        dist = defaultdict(int)
        for p in class_id_scores:
            dist[p[0]] += 1
        dist_list = []
        for k, v in dist.items():
            dist_list.append((k, v))
        new_dist_list = sorted(dist_list, key=lambda x: x[1], reverse=True)
        select_cls_id_according_to_the_most = new_dist_list[0][0]
        # Deal with conflict.
        if len(new_dist_list) > 1:
            if new_dist_list[0][1] == new_dist_list[1][1]:
                selected_cls_id_final = select_cls_id_according_to_max_score
            else:
                selected_cls_id_final = select_cls_id_according_to_the_most
        else:
            assert select_cls_id_according_to_max_score == select_cls_id_according_to_the_most
            selected_cls_id_final = select_cls_id_according_to_the_most
        # print(selected_cls_id_final, class_id_scores)
        result_list.append((obv_id, selected_cls_id_final))
    print(len(result_list))


    output_file = model_result_file + '.result.csv'
    with open(output_file, 'w') as wf:
        wf.write('observation_id,class_id\n')
        for it in result_list:
            wf.write('%s,%s\n' % (it[0], it[1]))
