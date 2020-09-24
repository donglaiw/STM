import os,sys
import shutil
import subprocess
from PIL import Image
import argparse
import glob
import numpy as np

import torch
import torch.nn as nn

from eval_DAVIS import Run_video
from model import STM
from helpers import overlay_davis

# 64 different colors
# avoid using matplotlib
colors=[[255,0,0],[0,255,0],[0,0,255], [255,255,0], [255,0,255],[0,255,255],[0,128,255],[128,0,255],[255,128,0],[0,255,128],[128,255,0],[255,0,128]]


def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("--input-index", type=str, help="cluster result",default='/local/DATA')
    parser.add_argument("--input-fps", type=int, help="input video fps",default=30)
    parser.add_argument("--image-step", type=int, help="image step for propagation",default=1)
    parser.add_argument("--image-template", type=str, help="image name template",default='/local/DATA')
    parser.add_argument("--mask-folder", type=str, help="mask folder",default='/local/DATA')
    parser.add_argument("--output-template", type=str, help="output mask name template",default='/local/DATA')
    parser.add_argument("--output-vis", type=int, help="output mask or visualization",default=0)
    parser.add_argument("--output-step", type=int, help="how many to output",default=0)
    parser.add_argument("--stm-height", type=int, help="image resize for propagation",default=480)
    parser.add_argument("--shot-chunk-len", type=int, help="max number of images for propagation",default=60)
    parser.add_argument("--stm-mem-step", type=int, help="memory step for propagation",default=1)
    parser.add_argument("--redo", type=int, help="overwrite previous results",default=0)
    return parser.parse_args()
def convertClusterStrToList(input_str):
    if input_str[-1] == ';':
        input_str = input_str[:-1]
    output_list = [np.array([int(y) for y in x.split(',')]) for x in input_str.split(';')]

    return output_list

def extractId(fn):
    err = 0
    offset = 0
    if '/' in fn:
        offset = fn.rfind('/')+1
        fn = fn[offset:]
    sid = 0
    while sid < len(fn) and (not fn[sid].isnumeric()):
        sid += 1
    if sid == len(fn)-1:
        err = 1
    else:
        lid = sid + 1
        while lid < len(fn) and fn[lid].isnumeric():
            lid += 1
    if err > 0:
        raise ValueError('No number found in ' + fn)
    return int(fn[sid:lid]), [offset+sid, offset+lid]

class YouTopDataLoader(object):
    def __init__(self, args):
        # image index: original video
        # mask index: either downsampled by fps or same as image index 
        # fps: frame per second
        # step: every K frame
        self.image_template = args.image_template
        self.output_template = args.output_template
        self.input_fps = args.input_fps
        self.image_step = args.image_step
        self.shot_list = convertClusterStrToList(args.input_index)
        self.shot_num = len(self.shot_list)
        # start from 0
        mask_files = sorted(glob.glob(args.mask_folder + '/*.png'))
        if len(mask_files) != self.shot_num: # exist empty frames
            sid, pos = extractId(mask_files[0])
            lid, _ = extractId(mask_files[-1])
            print('exist missing mask file: ',sid,lid,self.shot_num)
            mask_template = mask_files[0][:pos[0]] + '%0' + str(pos[1]-pos[0]) + 'd' + mask_files[0][pos[1]:]
            if lid - sid + 1 != self.shot_num: # assume start from 0
                lid = max(lid, self.shot_num-1)
                sid = lid-self.shot_num+1
            mask_files = [mask_template % x for x in range(sid,lid+1)]
        self.mask_files = mask_files 
        self.stm_height = args.stm_height
        self.output_step = args.output_step
        if self.output_step == 0:
            self.output_step = self.input_fps//self.image_step
        
        # make sure chunk_len aligns with input_fps + 1
        self.shot_chunk_len = (max(self.output_step, args.shot_chunk_len) // self.output_step) * self.output_step + 1
        self.getImageSize()

        output_folder = args.output_template[:args.output_template.rfind('/')]
        if not os.path.exists(output_folder):
            os.makedir(output_folder)

    def getImageSize(self):
        tmp_image_template = self.image_template % self.shot_list[0][0]
        tmp_image = Image.open(tmp_image_template)
        self.width, self.height = tmp_image.size  
        self.stm_width = int(self.width/float(self.height)*self.stm_height)


    def getShotAnchorLen(self, shot_index):
        return len(self.shot_list[shot_index])

    def getShotLen(self, shot_index):
        # don't do propagation for the last frame
        return (len(self.shot_list[shot_index]) - 1) * self.input_fps//self.image_step + 1

    def getShotChunkNum(self, shot_index):
        shot_len = self.getShotLen(shot_index)
        # 1 frame overlap between chunks
        chunk_len = self.shot_chunk_len - 1

        return 1 + (shot_len - self.shot_chunk_len + chunk_len -1) // chunk_len

    def getShotImageIndex(self, shot_index):
        image_index_anchor = self.shot_list[shot_index][:-1]
        image_index = image_index_anchor.reshape([-1,1]) + range(0, self.input_fps, self.image_step)
        # add the last frame
        self.image_index = np.array(list(image_index.ravel()) + [self.shot_list[shot_index][-1]])

    def All_to_onehot(self, masks, K = 1):
        Ms = np.zeros((K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for k in range(K):
            Ms[k] = (masks == k).astype(np.uint8)
        return Ms


    def getChunkStat(self, shot_index, chunk_index):
        chunk_start  = chunk_index * (self.shot_chunk_len - 1)
        chunk_len = min(self.shot_chunk_len, self.getShotLen(shot_index) - chunk_start)
        return chunk_start, chunk_len

    def getShotData(self, shot_index, chunk_index):
        chunk_start, chunk_len = self.getChunkStat(shot_index, chunk_index)
        N_masks = 255*np.ones([chunk_len, self.stm_height, self.stm_width], dtype=np.uint8)
        
        if chunk_index == 0:
            # read from input mask
            mask_file = self.mask_files[shot_index]
        else:
            # read from prop mask
            mask_file = self.output_template % self.image_index[chunk_start]
        if not os.path.exists(mask_file):
            print('%s does not exist!' % mask_file)
            return None, None, 0, None

        mask = Image.open(mask_file).convert('P')
        N_masks[0] = np.array(mask.resize((self.stm_width,self.stm_height),Image.NEAREST), dtype=np.uint8)
        # sometimes the mask ids are not range(K)
        # assume < 255 objects
        mask_id = np.unique(N_masks[0])
        mask_id_num = (mask_id>0).sum()
        mask_id_relabel = np.zeros(mask_id.max()+1, np.uint8)
        mask_id_relabel[mask_id[mask_id>0]] = range(1, 1+mask_id_num)
        mask_id_relabel[mask_id[mask_id>0]] = range(1, 1+mask_id_num)

        mask_id_relabel_inv = np.zeros(mask_id_num+1, np.uint8)
        mask_id_relabel_inv[1:] = mask_id[mask_id>0]
        N_masks[0] = mask_id_relabel[N_masks[0]]

        K = N_masks[0].max()
        if K == 0:# no mask to propagate
            return None, None, K, None

        N_frames = np.empty([chunk_len, self.stm_height, self.stm_width, 3], dtype=np.float32)
        for f in range(chunk_len):
            index = self.image_index[chunk_start + f]
            image_file = self.image_template % index
            image = Image.open(image_file).convert('RGB')
            N_frames[f] = np.array(image.resize((self.stm_width,self.stm_height),Image.BILINEAR))/255.
        
        # add the extra batch dimension
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()[None,:]).float()
        if K == 1:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
        Ms = torch.from_numpy(self.All_to_onehot(N_masks, K+1).copy()[None,:]).float()
        num_objects = torch.LongTensor([K])

        return Fs, Ms, num_objects, mask_id_relabel_inv

    def getShotOutputIndex(self, shot_index, chunk_index):
        chunk_start, chunk_len = self.getChunkStat(shot_index, chunk_index)
        result_index = np.arange(self.output_step, chunk_len, self.output_step)
        return result_index, self.image_index[chunk_start + result_index]

if __name__ == '__main__' :
    # load model
    model = nn.DataParallel(STM())
    if torch.cuda.is_available():
        model.cuda()
    model.eval() # turn-off BN
    pth_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'STM_weights.pth')
    model.load_state_dict(torch.load(pth_path))

    # load data
    args = get_arguments()
    dataloader = YouTopDataLoader(args)
    # util
    for shot_id in range(dataloader.shot_num):
        print('shot',shot_id)
        dataloader.getShotImageIndex(shot_id)
        # copy first frame mask
        mask_file_out = dataloader.output_template % dataloader.image_index[0]
        if args.redo or not os.path.exists(mask_file_out):
            mask_file_in = dataloader.mask_files[shot_id]
            if os.path.exists(mask_file_in):
                shutil.copyfile(mask_file_in, mask_file_out)
            else:
                output = Image.fromarray(np.zeros([dataloader.height, dataloader.width], np.uint8))
                output.save(mask_file_out)

        anchor_num = dataloader.getShotAnchorLen(shot_id)
        if anchor_num == 1: # only one anchor frame, no need to propagate
            continue
        chunk_num = dataloader.getShotChunkNum(shot_id)
        for chunk_id in range(chunk_num):
            result_id, output_id = dataloader.getShotOutputIndex(shot_id, chunk_id)
            if args.redo or not os.path.exists(dataloader.output_template % output_id[-1]):
                Fs, Ms, num_objects, mask_id_relabel_inv = dataloader.getShotData(shot_id, chunk_id)
                if num_objects > 0:
                    print(chunk_id, num_objects)
                    pred, Es = Run_video(model, Fs, Ms, Fs.shape[2], num_objects,\
                                         Mem_every = args.stm_mem_step, Mem_number=None)

                    pred = mask_id_relabel_inv[pred]
                    for z in range(len(result_id)):
                        if args.output_vis == 0:
                            output = Image.fromarray(pred[result_id[z]])
                        else:
                            pF = (Fs[0,:,result_id[z]].permute(1,2,0).numpy() * 255.).astype(np.uint8)
                            pE = pred[result_id[z]]
                            output = Image.fromarray(overlay_davis(pF, pE, colors))
                        output = output.resize((dataloader.width, dataloader.height),Image.NEAREST)
                        output.save(dataloader.output_template % output_id[z])
