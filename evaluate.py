import sys
sys.path.append('core')
import os
os.environ["KMP_BLOCKTIME"] = "0"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from utils.utils import InputPadder
from matplotlib import colors
import numpy as np
import imageio
# import flow_vis

def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
#     flow = flow[0, :, :, :]
    flow_np = flow.detach().cpu().numpy()
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow_np.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow_np).flatten()
    data.tofile(f)
    f.close() 

def reduce_list(lists, nprocs):
    new_lists = {}
    for key, value in lists.items():
        rt = value.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        new_lists[key] = rt.item()
    return new_lists


def reduce_tensor(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def evaluates(args, model, datasets, names, metric_fun, logger=None):
    if logger is not None:
        _print = logger.log_info
    else:
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    metrics = {}
    for val_set, name in zip(datasets, names):
        if args.distributed == 'ddp':
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, \
                seed=args.seed, drop_last=False)
        else:
            val_sampler = None
        val_loader = DataLoader(val_set, args.test_batch_size, num_workers=args.jobs, sampler=val_sampler)
        if args.distributed != 'ddp' or args.local_rank == 0:
            _print(">>> For evaluate {}, use length (bs/loader/set): ({}/{}/{})".format( \
                name, args.test_batch_size, len(val_loader), len(val_set)), "evaluates")
        metric = evaluate(args, model, val_loader, name, metric_fun, logger=logger)

        for key, values in metric.items():
            new_key = "val_{}/{}".format(name, key)
            assert new_key not in metrics
            metrics[new_key] = values

    return metrics

def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

def evaluate(args, model, dataloader, name, metric_fun, logger=None):
    if logger is not None:
        _print = logger.log_info
    else:
        def print_line(line, subname=None):
            print(line)
        _print = print_line

    start = time.time()
    model.eval()

    metric_fun.clear()

    if args.distributed != 'ddp' or args.local_rank == 0:
        bar = tqdm(total=len(dataloader), position=0, leave=True)

    for index, batch in enumerate(dataloader):

        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].cuda(args.gpus[args.local_rank] \
                    if args.local_rank != -1 else 0, non_blocking=True)

        padder = InputPadder(batch['image1'].shape, div=args.pad)
        pad_batch = padder.pad_batch(batch)

        torch.cuda.synchronize()
        tm = time.time()

        with torch.no_grad():
            output = model(pad_batch, iters=args.iters)

        torch.cuda.synchronize()
        elapsed = time.time() - tm

        output['flow_pred'] = padder.unpad(output['flow_final'])
#         print(batch['flow_gt'])
#################################
#         print("output['flow_pred'].shape:",output['flow_pred'].shape)  # torch.Size([1, 2, 720, 1280])
        flo = output['flow_pred'][0].permute(1, 2, 0).cpu().numpy()
        dst_path = "/research/DCEIFlow/result_gopro_adjtemp/"
        if not os.path.exists(os.path.join(dst_path, "flow_flo")):
            os.makedirs(os.path.join(dst_path, "flow_flo"))
#         print("check jinjin", batch['basename'])
#         print("check jinjin", batch['basename'][index])
        out_path = os.path.join(dst_path, "flow_flo", batch['basename'][0]+'.flo')

        write_flo(output['flow_pred'][0].permute(1, 2, 0), out_path)
        
        uv = flo * 128.0 + 2**15
        valid = np.ones([uv.shape[0], uv.shape[1], 1])
        uv = np.concatenate([uv, valid], axis=-1).astype(np.uint8)

        dst_path = "/research/DCEIFlow/result_gopro_adjtemp/"
        if not os.path.exists(os.path.join(dst_path, "flows")):
            os.makedirs(os.path.join(dst_path, "flows"))
        out_path = os.path.join(dst_path, "flows", batch['basename'][0]+'.png')
#         print("out_path:",out_path)
        imageio.imwrite(out_path, uv)#, format='PNG-FI')


#         test_save = args.test_save
#         scene = os.path.join(test_save, scene)
#         if not os.path.exists(scene):
#             os.makedirs(scene)
#         path_to_file = os.path.join(scene, ind+'.png')
# #         path_to_file = os.path.join(test_save, ind+'.png')
#         imageio.imwrite(path_to_file, uv, format='PNG-FI')
        flo[np.isinf(flo)] = 0
#         flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
        flow_color = flow2img(flo)

#         # flow -> numpy array 2 x height x width
#         # 2,h,w -> h,w,2
# #         flow = flow.transpose(1,2,0)
# #         flow[numpy.isinf(flow)]=0
#         # Use Hue, Saturation, Value colour model
#         hsv = np.zeros((flo.shape[0], flo.shape[1], 3), dtype=float)

#         # The additional **0.5 is a scaling factor
#         mag = np.sqrt(flo[...,0]**2+flo[...,1]**2)**0.5

#         ang = np.arctan2(flo[...,1], flo[...,0])
#         ang[ang<0]+=np.pi*2
#         hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
#         hsv[..., 1] = 1
# #         if scaling is None:
#         hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
#         rgb = colors.hsv_to_rgb(hsv)
#         # This all seems like an overkill, but it's just to exactly match the cv2 implementation
#         bgr = np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)

#         out = bgr*255
        dst_path = "/research/DCEIFlow/result_gopro_adjtemp/"
        if not os.path.exists(os.path.join(dst_path, "visualization")):
            os.makedirs(os.path.join(dst_path, "visualization"))
        out_path = os.path.join(dst_path, "visualization", batch['basename'][0]+'.png')
#         print("out_path:",out_path)
        imageio.imwrite(out_path, flow_color.astype('uint8'))#, format='PNG-FI')

        dst_path = "/research/DCEIFlow/result_gopro_adjtemp/"
        if not os.path.exists(os.path.join(dst_path, "images")):
            os.makedirs(os.path.join(dst_path, "images"))
        out_path = os.path.join(dst_path, "images", batch['basename'][0]+'.png')
        imageio.imwrite(out_path, batch['image1'][0].permute(1, 2, 0).cpu().numpy().astype('uint8'))#, format='PNG-FI')
        
        flo = batch['flow_gt'][0].permute(1, 2, 0).cpu().numpy()
        flo[np.isinf(flo)] = 0
#         flow_color = flow_vis.flow_to_color(flo, convert_to_bgr=False)
        flow_color = flow2img(flo)
#         flo = batch['flow_gt'][0].permute(1, 2, 0).cpu().numpy()
#         flo[np.isinf(flo)] = 0
#         # flow -> numpy array 2 x height x width
#         # 2,h,w -> h,w,2
# #         flow = flow.transpose(1,2,0)
# #         flow[numpy.isinf(flow)]=0
#         # Use Hue, Saturation, Value colour model
#         hsv = np.zeros((flo.shape[0], flo.shape[1], 3), dtype=float)

#         # The additional **0.5 is a scaling factor
#         mag = np.sqrt(flo[...,0]**2+flo[...,1]**2)**0.5

#         ang = np.arctan2(flo[...,1], flo[...,0])
#         ang[ang<0]+=np.pi*2
#         hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
#         hsv[..., 1] = 1
# #         if scaling is None:
#         hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
#         rgb = colors.hsv_to_rgb(hsv)
#         # This all seems like an overkill, but it's just to exactly match the cv2 implementation
#         bgr = np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)

#         out = bgr*255
        dst_path = "/research/DCEIFlow/result_gopro_adjtemp/"
        if not os.path.exists(os.path.join(dst_path, "visualization_gt")):
            os.makedirs(os.path.join(dst_path, "visualization_gt"))
        out_path = os.path.join(dst_path, "visualization_gt", batch['basename'][0]+'.png')
#         print("out_path:",out_path)
        imageio.imwrite(out_path, flow_color.astype('uint8'))#, format='PNG-FI')
#         if not os.path.exists(os.path.join(scene, "visualization")):
#             os.makedirs(os.path.join(scene, "visualization"))
#         path_to_file = os.path.join(scene, "visualization", ind+'_visualize.png')
#         imageio.imwrite(path_to_file, out.astype('uint8'), format='PNG-FI')
#################################
        if args.isbi and 'flow_final_bw' in output.keys():
            output['flow_pred_bw'] = padder.unpad(output['flow_final_bw'])

        if 'image1_valid' in batch.keys():
            output['flow_pred'][batch['image1_valid'].repeat(1, 2, 1, 1) < 0.5] = 0

        metric_each = metric_fun.calculate(output, batch, name)

        if args.distributed == 'ddp':
            torch.distributed.barrier()
            reduced_metric_each = reduce_list(metric_each, args.nprocs)
        else:
            reduced_metric_each = metric_each

        reduced_metric_each.update({'time': elapsed})

        if args.distributed != 'ddp' or args.local_rank == 0:
            metric_fun.push(reduced_metric_each)

        if args.distributed != 'ddp' or args.local_rank == 0:
            if 'masked_epe' in metric_each.keys():
                bar.set_description("{}/{}[{}:{}], time:{:8.6f}, epe:{:8.6f}, masked_epe:{:8.6f}".format(index * len(batch['basename']), \
                    len(dataloader.dataset), batch['raw_index'][0], batch['basename'][0], elapsed, metric_each['epe'], metric_each['masked_epe']))
            else:
                bar.set_description("{}/{}[{}:{}],time:{:8.6f}, epe:{:8.6f}".format(index * len(batch['basename']), \
                    len(dataloader.dataset), batch['raw_index'][0], batch['basename'][0], elapsed, metric_each['epe']))
            bar.update(1)

    if args.distributed != 'ddp' or args.local_rank == 0:
        bar.close()
    metrics_str, all_metrics = metric_fun.summary()
    metric_fun.clear()

    if args.distributed != 'ddp' or args.local_rank == 0:
        _print("<<< In {} eval: {} (100X F1), with time {}s.".format(name, metrics_str, time.time() - start), "evaluate")

    model.train()
    return all_metrics
