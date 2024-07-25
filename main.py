import sys
sys.path.append('core')
import os
os.environ["KMP_BLOCKTIME"] = "0"
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from logger import Logger
from argparser import ArgParser
from trainer import Trainer
from evaluate import evaluates
from utils.datasets import fetch_dataset, fetch_test_dataset
from utils.utils import setup_seed, count_parameters, count_all_parameters, build_module
from tqdm import tqdm
import imageio
from utils.utils import InputPadder
from matplotlib import colors
import flow_vis

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy gradscale for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def train(local_rank, args):

    args.local_rank = local_rank

    if args.local_rank == 0:
        logger = Logger(args)
        for k, v in vars(args).items():
            logger.log_debug('{}\t=\t{}'.format(k, v), "argparser")
        _print = logger.log_info
    else:
        logger = None
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    if args.distributed == 'ddp':
        dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(args.ip, args.port), world_size=args.nprocs, rank=local_rank)
        torch.cuda.set_device(args.local_rank)

    train_set, val_sets, val_setnames = fetch_dataset(args)
    if args.distributed == 'ddp':
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=args.seed, drop_last=False)
        train_loader = DataLoader(train_set, args.batch_size, num_workers=args.jobs, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.jobs, \
            pin_memory=True, drop_last=True, sampler=None)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use training set {} with length: bs/loader/dataset ({}/{}({})/{})".format( \
            args.stage, args.batch_size, len(train_loader), len(train_loader.dataset), len(train_set)))

    assert len(val_setnames) == len(val_sets)
    val_length_str = ""
    for val_set, name in zip(val_sets, val_setnames):
        val_length_str += "({}/{}),".format(name, len(val_set))
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use validation set: test_bs={}, name/datalength:{}".format( \
            args.test_batch_size, val_length_str))

    model = build_module("core", args.model)(args)
    if args.distributed == 'ddp':
        model.cuda(args.gpus[args.local_rank])
        model.train()
        model = torch.nn.parallel.DistributedDataParallel(model, \
            device_ids=[args.gpus[args.local_rank]])
        _print("Use DistributedDataParallel at gpu {} with find_unused_parameters:False".format( \
            args.gpus[args.local_rank]))
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
        model.cuda(args.gpus[0])
        model.train()

    loss = build_module("core.loss", "Combine")(args)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use losses: {} with weights: {}".format(args.loss, args.loss_weights))

    metric_fun = build_module("core.metric", "Combine")(args)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use metrics: {}".format(args.metric))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, \
        eps=args.epsilon)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use optimizer: {} with init lr:{}, decay:{}, epsilon:{} ".format( \
            "AdamW", args.lr, args.weight_decay, args.epsilon))

    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, \
        steps_per_epoch=len(train_loader), epochs=args.epoch)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use scheduler: {}, with epoch:{}, steps_per_epoch {}".format( \
            "OneCycleLR", args.epoch, len(train_loader)))

    scaler = GradScaler(enabled=args.mixed_precision)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use gradscaler with mixed_precision? {}".format(args.mixed_precision))

    trainer = Trainer(args, model, loss=loss, optimizer=optimizer, \
        lr_scheduler=lr_scheduler, scaler=scaler, logger=logger)
    start = 0
    if args.checkpoint != '':
        start = trainer.load(args.checkpoint, only_model=False if args.resume else True)

    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("For model {} with name {}, Parameter Count: {}(trainable)/{}(all), gpus: {}".format( \
            args.model, args.name if args.name != "" else "NoNAME", count_parameters(trainer.model), \
                count_all_parameters(trainer.model), args.gpus))
        _print("Use small? {}".format(args.small))

    setup_seed(args.seed)

    for i in range(start+1, args.epoch+1):
        if args.distributed != 'ddp' or args.local_rank == 0:
            _print(">>> Start the {}/{} training epoch with save feq {} at stage {}".format( \
                i, args.epoch, args.save_feq, args.stage), "training")
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        trainer.run_epoch(train_loader)
        if args.local_rank == 0 and logger is not None:
            logger.summary(i)

        if i % args.eval_feq == 0:
            if args.distributed != 'ddp' or args.local_rank == 0:
                _print(">>> Run {} evaluate epoch".format(i), "training")
            scores = evaluates(args, model, val_sets, val_setnames, metric_fun, logger=logger)
            if args.local_rank == 0 and logger is not None:
                logger.write_dict(i, scores)
        if args.local_rank == 0 and i % args.save_feq == 0:
            trainer.store(args.save_path, args.name, i)

    if args.local_rank == 0:
        dist.destroy_process_group()
        _print("Destroy_process_group", 'train')

    if logger is not None:
        logger.close()


def test(local_rank, args, logger=None):

    args.local_rank = local_rank

    if logger is not None:
        _print = logger.log_info
    else:
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    if args.distributed == 'ddp':
        dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(args.ip, args.port), world_size=args.nprocs, rank=local_rank)
        torch.cuda.set_device(args.local_rank)

    assert args.checkpoint != ''

    start = time.time()
    test_sets, test_setnames = fetch_test_dataset(args)

    assert len(test_setnames) == len(test_sets)
    test_length_str = ""
    for test_set, name in zip(test_sets, test_setnames):
        test_length_str += "({}/{}),".format(name, len(test_set))
    
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use test set: test_bs={}, name/datalength:{}".format(args.test_batch_size, test_length_str), 'test')

    metric_fun = build_module("core.metric", "Combine")(args)
    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("Use metrics: {}".format(args.metric), 'test')

    model = build_module("core", args.model)(args)

    if args.checkpoint != '':
        if args.distributed != 'ddp' or args.local_rank == 0:
            _print("Evalulate Model {} for checkpoint {}".format(args.model, args.checkpoint), 'test')
            _print("For model {} with name {}, Parameter Count: {}(trainable)/{}(all), gpus: {}".format( \
                args.model, args.name if args.name != "" else "NoNAME", count_parameters(model), \
                    count_all_parameters(model), args.gpus))

        state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        try:
            if "model" in state_dict.keys():
                state_dict = state_dict.pop("model")
            elif 'model_state_dict' in state_dict.keys():
                state_dict = state_dict.pop("model_state_dict")

            if "module." in list(state_dict.keys())[0]:
                for key in list(state_dict.keys()):
                    state_dict.update({key[7:]:state_dict.pop(key)})

            model.load_state_dict(state_dict)
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(args.checkpoint))
    else:
        raise NotImplementedError("Please set --checkpoint")

    if args.distributed == 'ddp':
        model.cuda(args.local_rank)
        model.eval()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpus[local_rank]])
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
        model.cuda(args.gpus[0])
        model.eval()

    scores = evaluates(args, model, test_sets, test_setnames, metric_fun, logger=logger)

    summary_str = ""
    for key in scores.keys():
        summary_str += "({}/{}),".format(key, scores[key])

    if args.distributed != 'ddp' or args.local_rank == 0:
        dist.destroy_process_group()
        _print("Destroy_process_group", 'test')

        _print("Test complete, {}, time consuming {}/s".format(summary_str, time.time() - start), 'test')
        
# def inference(local_rank, args, logger=None):

#     args.local_rank = local_rank

#     if logger is not None:
#         _print = logger.log_info
#     else:
#         def print_line(line, subname=None):
#             print(line)
#         _print = print_line

#     if args.distributed == 'ddp':
#         dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(args.ip, args.port), world_size=args.nprocs, rank=local_rank)
#         torch.cuda.set_device(args.local_rank)

#     assert args.checkpoint != ''

#     start = time.time()
#     test_sets, test_setnames = fetch_test_dataset(args)

#     assert len(test_setnames) == len(test_sets)
#     test_length_str = ""
#     for test_set, name in zip(test_sets, test_setnames):
#         test_length_str += "({}/{}),".format(name, len(test_set))
    
#     if args.distributed != 'ddp' or args.local_rank == 0:
#         _print("Use test set: test_bs={}, name/datalength:{}".format(args.test_batch_size, test_length_str), 'test')

# #     metric_fun = build_module("core.metric", "Combine")(args)
# #     if args.distributed != 'ddp' or args.local_rank == 0:
# #         _print("Use metrics: {}".format(args.metric), 'test')

#     model = build_module("core", args.model)(args)

#     if args.checkpoint != '':
#         if args.distributed != 'ddp' or args.local_rank == 0:
#             _print("Evalulate Model {} for checkpoint {}".format(args.model, args.checkpoint), 'test')
#             _print("For model {} with name {}, Parameter Count: {}(trainable)/{}(all), gpus: {}".format( \
#                 args.model, args.name if args.name != "" else "NoNAME", count_parameters(model), \
#                     count_all_parameters(model), args.gpus))

#         state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))
#         try:
#             if "model" in state_dict.keys():
#                 state_dict = state_dict.pop("model")
#             elif 'model_state_dict' in state_dict.keys():
#                 state_dict = state_dict.pop("model_state_dict")

#             if "module." in list(state_dict.keys())[0]:
#                 for key in list(state_dict.keys()):
#                     state_dict.update({key[7:]:state_dict.pop(key)})

#             model.load_state_dict(state_dict)
#         except:
#             raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(args.checkpoint))
#     else:
#         raise NotImplementedError("Please set --checkpoint")

#     if args.distributed == 'ddp':
#         model.cuda(args.local_rank)
#         model.eval()
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpus[local_rank]])
#     else:
#         model = nn.DataParallel(model, device_ids=args.gpus)
#         model.cuda(args.gpus[0])
#         model.eval()

# #     if logger is not None:
# #         _print = logger.log_info
# #     else:
# #         def print_line(line, subname=None):
# #             print(line)
# #         _print = print_line

# #     metrics = {}

# #     print("test_setnames",test_setnames)
# #     print("test_sets",test_sets)
    
#     for val_set, name in zip(test_sets, test_setnames):
# #         print("val_set", val_set)  # val_set <utils.datasets.GoPro.GoPro object at 0x7f07a6d65d30>
# #         print("name", name)  # name GOPR0384_11_00
#         if args.distributed == 'ddp':
#             val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, \
#                 seed=args.seed, drop_last=False)
#         else:
#             val_sampler = None
#         val_loader = DataLoader(val_set, args.test_batch_size, num_workers=args.jobs, sampler=val_sampler)
#         if args.distributed != 'ddp' or args.local_rank == 0:
#             _print(">>> For evaluate {}, use length (bs/loader/set): ({}/{}/{})".format( \
#                 name, args.test_batch_size, len(val_loader), len(val_set)), "evaluates")
#         #metric = evaluate(args, model, val_loader, name, metric_fun, logger=logger)

# #         for key, values in metric.items():
# #             new_key = "val_{}/{}".format(name, key)
# #             assert new_key not in metrics
# #             metrics[new_key] = values
#     start = time.time()
#     model.eval()

# #     metric_fun.clear()

#     if args.distributed != 'ddp' or args.local_rank == 0:
#         bar = tqdm(total=len(val_loader), position=0, leave=True)

#     for index, batch in enumerate(val_loader):

#         for key in batch.keys():
#             if torch.is_tensor(batch[key]):
#                 batch[key] = batch[key].cuda(args.gpus[args.local_rank] \
#                     if args.local_rank != -1 else 0, non_blocking=True)

#         padder = InputPadder(batch['image1'].shape, div=args.pad)
#         pad_batch = padder.pad_batch(batch)
# #         print("pad_batch\n",pad_batch)
# #         print("pad_batch['image1'].shape:",pad_batch['image1'].shape)  # pad_batch['image1'].shape: torch.Size([1, 3, 720, 1280])
# #         print("pad_batch['event_voxel'].shape:",pad_batch['event_voxel'].shape)  # pad_batch['event_voxel'].shape: torch.Size([1, 10, 720, 1280])
# #         print("pad_batch['event_valid'].shape:",pad_batch['event_valid'].shape)  # pad_batch['event_valid'].shape: torch.Size([1, 1, 720, 1280])
        
#         torch.cuda.synchronize()
#         tm = time.time()

#         with torch.no_grad():
#             output = model(pad_batch, iters=args.iters)
        
#         torch.cuda.synchronize()
#         elapsed = time.time() - tm

#         output['flow_pred'] = padder.unpad(output['flow_final'])
#         if args.isbi and 'flow_final_bw' in output.keys():
#             output['flow_pred_bw'] = padder.unpad(output['flow_final_bw'])

#         if 'image1_valid' in batch.keys():
#             output['flow_pred'][batch['image1_valid'].repeat(1, 2, 1, 1) < 0.5] = 0

# #################################
# #         print("output['flow_pred'].shape:",output['flow_pred'].shape)  # torch.Size([1, 2, 720, 1280])
#         flo = output['flow_pred'][0].permute(1, 2, 0).cpu().numpy()
        
#         uv = flo * 128.0 + 2**15
#         valid = np.ones([uv.shape[0], uv.shape[1], 1])
#         uv = np.concatenate([uv, valid], axis=-1).astype(np.uint8)

#         dst_path = "/research/DCEIFlow/result_gopro2_train/"
#         if not os.path.exists(os.path.join(dst_path, "flows")):
#             os.makedirs(os.path.join(dst_path, "flows"))
#         out_path = os.path.join(dst_path, "flows", str(index).zfill(6)+'.png')
# #         print("out_path:",out_path)
#         imageio.imwrite(out_path, uv)#, format='PNG-FI')


# #         test_save = args.test_save
# #         scene = os.path.join(test_save, scene)
# #         if not os.path.exists(scene):
# #             os.makedirs(scene)
# #         path_to_file = os.path.join(scene, ind+'.png')
# # #         path_to_file = os.path.join(test_save, ind+'.png')
# #         imageio.imwrite(path_to_file, uv, format='PNG-FI')
#         dst_path = "/research/DCEIFlow/result_gopro2_train/"
#         if not os.path.exists(os.path.join(dst_path, "images")):
#             os.makedirs(os.path.join(dst_path, "images"))
#         out_path = os.path.join(dst_path, "images", str(index).zfill(6)+'.png')
#         imageio.imwrite(out_path, batch['image1'][0].permute(1, 2, 0).cpu().numpy().astype('uint8'))#, format='PNG-FI')

#         flo[np.isinf(flo)] = 0
#         flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
# #         # flow -> numpy array 2 x height x width
# #         # 2,h,w -> h,w,2
# # #         flow = flow.transpose(1,2,0)
# # #         flow[numpy.isinf(flow)]=0
# #         # Use Hue, Saturation, Value colour model
# #         hsv = np.zeros((flo.shape[0], flo.shape[1], 3), dtype=float)

# #         # The additional **0.5 is a scaling factor
# #         mag = np.sqrt(flo[...,0]**2+flo[...,1]**2)**0.5

# #         ang = np.arctan2(flo[...,1], flo[...,0])
# #         ang[ang<0]+=np.pi*2
# #         hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
# #         hsv[..., 1] = 1
# # #         if scaling is None:
# #         hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
# #         rgb = colors.hsv_to_rgb(hsv)
# #         # This all seems like an overkill, but it's just to exactly match the cv2 implementation
# #         bgr = np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)

# #         out = bgr*255
#         dst_path = "/research/DCEIFlow/result_gopro2_train/"
#         if not os.path.exists(os.path.join(dst_path, "visualization")):
#             os.makedirs(os.path.join(dst_path, "visualization"))
#         out_path = os.path.join(dst_path, "visualization", str(index).zfill(6)+'.png')
# #         print("out_path:",out_path)
#         imageio.imwrite(out_path, flow_color.astype('uint8'))#, format='PNG-FI')
# #         if not os.path.exists(os.path.join(scene, "visualization")):
# #             os.makedirs(os.path.join(scene, "visualization"))
# #         path_to_file = os.path.join(scene, "visualization", ind+'_visualize.png')
# #         imageio.imwrite(path_to_file, out.astype('uint8'), format='PNG-FI')
# #################################

#         bar.update(1)
# #         metric_each = metric_fun.calculate(output, batch, test_setnames)

# #         if args.distributed == 'ddp':
# #             torch.distributed.barrier()
# #             reduced_metric_each = reduce_list(metric_each, args.nprocs)
# #         else:
# #             reduced_metric_each = metric_each

# #         reduced_metric_each.update({'time': elapsed})

# #         if args.distributed != 'ddp' or args.local_rank == 0:
# #             metric_fun.push(reduced_metric_each)

# #         if args.distributed != 'ddp' or args.local_rank == 0:
# #             if 'masked_epe' in metric_each.keys():
# #                 bar.set_description("{}/{}[{}:{}], time:{:8.6f}, epe:{:8.6f}, masked_epe:{:8.6f}".format(index * len(batch['basename']), \
# #                     len(val_loader.dataset), batch['raw_index'][0], batch['basename'][0], elapsed, metric_each['epe'], metric_each['masked_epe']))
# #             else:
# #                 bar.set_description("{}/{}[{}:{}],time:{:8.6f}, epe:{:8.6f}".format(index * len(batch['basename']), \
# #                     len(val_loader.dataset), batch['raw_index'][0], batch['basename'][0], elapsed, metric_each['epe']))
# #             bar.update(1)

#     if args.distributed != 'ddp' or args.local_rank == 0:
#         bar.close()
# #     metrics_str, all_metrics = metric_fun.summary()
# #     metric_fun.clear()

# #     if args.distributed != 'ddp' or args.local_rank == 0:
# #         _print("<<< In {} eval: {} (100X F1), with time {}s.".format(test_setnames, metrics_str, time.time() - start), "evaluate")

#     model.train()

#     scores = evaluates(args, model, test_sets, test_setnames, metric_fun, logger=logger)

#     summary_str = ""
#     for key in scores.keys():
#         summary_str += "({}/{}),".format(key, scores[key])

#     if args.distributed != 'ddp' or args.local_rank == 0:
#         dist.destroy_process_group()
#         _print("Destroy_process_group", 'test')

#         _print("Test complete, {}, time consuming {}/s".format(summary_str, time.time() - start), 'test')


if __name__ == "__main__":
    argparser = ArgParser()
    args = argparser.parser()
    setup_seed(args.seed)

    if args.gpus[0] == -1:
        args.gpus = [i for i in range(1)]#torch.cuda.device_count())]
    args.nprocs = len(args.gpus)

    if args.task == "train":
        if args.distributed == 'ddp':
            mp.spawn(train, nprocs=args.nprocs, args=(args, ))
        else:
            train(-1, args)
    elif args.task[:4] == "test":
        if args.distributed == 'ddp':
            mp.spawn(test, nprocs=args.nprocs, args=(args, ))
        else:
            train(-1, args)
#     elif args.task == "inference":
#         if args.distributed == 'ddp':
#             mp.spawn(inference, nprocs=args.nprocs, args=(args, ))
#         else:
#             train(-1, args)
    else:
        print("task error")
