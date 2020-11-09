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

# avoid using matplotlib
colors=[[255,0,0],[0,255,0],[0,0,255], [255,255,0], [255,0,255],[0,255,255],[0,128,255],[128,0,255],[255,128,0],[0,255,128],[128,255,0],[255,0,128]]


def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("--video-fps", type=int, help="input video fps",default=30)
    parser.add_argument("--image-template", type=str, help="image name template",default='/local/DATA')
    parser.add_argument("--image-cluster", type=str, help="cluster result",default='/local/DATA')
    parser.add_argument("--mask-folder", type=str, help="mask folder",default='/local/DATA')
    parser.add_argument("--mask-index-factor", type=str, help="need to x*k+b for input mask index",default='1,0')
    parser.add_argument("--mask-index-factor-output", type=str, help="need to (x-b)/k for output mask index",default='1,0')
    parser.add_argument("--mask-template-output", type=str, help="output mask name template",default='/local/DATA')
    parser.add_argument("--output-vis", type=int, help="output mask or visualization",default=0)
    parser.add_argument("--stm-height", type=int, help="image resize for propagation",default=480)
    parser.add_argument("--stm-step", type=int, help="image step for propagation",default=1)
    parser.add_argument("--stm-mem-len", type=int, help="max number of images for propagation",default=60)
    parser.add_argument("--stm-mem-step", type=int, help="memory step for propagation",default=1)
    parser.add_argument("--stm-anchor-num", type=int, help="number of anchor to use for stm",default=-1)
    parser.add_argument("--redo", type=int, help="overwrite previous results",default=0)
    args = parser.parse_args()
    args.mask_index_factor = [int(x) for x in args.mask_index_factor.split(',')]
    args.mask_index_factor_output = [int(x) for x in args.mask_index_factor_output.split(',')]
    return args

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

def relabelMask(mask):
    mask_id = np.unique(mask)
    mask_id_num = (mask_id>0).sum()
    mask_id_relabel = np.zeros(mask_id.max()+1, np.uint8)
    mask_id_relabel[mask_id[mask_id>0]] = range(1, 1+mask_id_num)

    mask_id_relabel_inv = np.zeros(mask_id_num+1, np.uint8)
    mask_id_relabel_inv[1:] = mask_id[mask_id>0]
    return mask_id_relabel, mask_id_relabel_inv

def visualizeMask(im, mask, template_output='./db/test%d.png'):
    for z in range(im.shape[0]):
        pF = (im[z] * 255.).astype(np.uint8)
        pE = mask[z]
        output = Image.fromarray(overlay_davis(pF, pE, colors))
        output.save(template_output % z)

def removeArr(arr1, arr2):
    return arr1[np.in1d(arr1, arr2, invert=True)]

class YouTopDataLoader(object):
    def __init__(self, args):
        # image index: original video
        # mask index: either downsampled by fps or same as image index 
        # fps: frame per second
        # step: every K frame

        # shot info
        self.shot_list = convertClusterStrToList(args.image_cluster)
        self.shot_num = len(self.shot_list)

        # image info
        self.image_template = args.image_template
        self.video_fps = args.video_fps
        self.stm_step = args.stm_step
        self.stm_height = args.stm_height
        self.width, self.height, self.stm_width = self.getImageSize() 

        # output
        self.mask_template_output = args.mask_template_output
        self.mask_index_factor = args.mask_index_factor
        self.mask_index_factor_output = args.mask_index_factor_output
        output_folder = args.mask_template_output[:args.mask_template_output.rfind('/')]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # mask info
        self.stm_mode = 'index'
        if self.mask_index_factor[0] == 0:
            # If the mask_index is only the index for the image cluster
            self.stm_mode ='shot' 
        self.mask_ids, self.mask_files = self.getMaskInfo()

        # for stm_mode=shot: make sure chunk_len aligns with video_fps + 1, so that the last prediction can be saved
        # for efficiency
        # how many steps needed for one output mask
        stm_step_num = self.mask_index_factor_output[0]//self.stm_step
        self.stm_mem_len = (max(stm_step_num, args.stm_mem_len) // stm_step_num) * stm_step_num + 1
        self.stm_anchor_num = args.stm_anchor_num

    def getMaskInfo(self):
        # start from 0
        mask_files = sorted(glob.glob(args.mask_folder + '/*.png'))
        if len(mask_files) == 0:
            sys.exit('no reference mask provided')
        # recover original corresponding image ids
        if self.stm_mode == 'shot':
            # no change index, assume shot-dependent
            # no need to match id
            mask_ids = np.arange(self.shot_num)
            if len(mask_files) != self.shot_num: # exist empty frames
                sid, pos = extractId(mask_files[0])
                lid, _ = extractId(mask_files[-1])
                print('exist missing mask file: ',sid,lid,self.shot_num)
                if lid - sid + 1 != self.shot_num: # assume start from 0
                    lid = max(lid, self.shot_num-1)
                    sid = lid-self.shot_num+1
                mask_ids = np.arange(sid,lid+1)
                mask_template = mask_files[0][:pos[0]] + '%0' + str(pos[1]-pos[0]) + 'd' + mask_files[0][pos[1]:]
                mask_files = [mask_template % x for x in mask_ids]
            mask_ids = [self.shot_list[x][0] for x in mask_ids]
        elif self.stm_mode == 'index':
            # 1fps -> 6 fps
            mask_ids = np.zeros(len(mask_files), int)
            for i in range(len(mask_files)):
                mask_ids[i], _ = extractId(mask_files[i])
            mask_ids = mask_ids * self.mask_index_factor[0] + self.mask_index_factor[1]
        return mask_ids, mask_files

    def convertIndexOutput(self, ind):
        return (ind - self.mask_index_factor_output[1]) // self.mask_index_factor_output[0]

    def getImageSize(self):
        tmp_image_template = self.image_template % self.shot_list[0][0]
        tmp_image = Image.open(tmp_image_template)
        width, height = tmp_image.size
        stm_width = int(width/float(height)*self.stm_height)
        return width, height, stm_width 

    def getShotAnchorLen(self, shot_index):
        return len(self.shot_list[shot_index])

    def getShotChunkNum(self, shot_index):
        if self.stm_anchor_num == -1: 
            # use all anchors
            shot_len = len(self.image_index)
            # The length for every chunk is chunk_len
            chunk_len = self.stm_mem_len - len(self.mask_index)
            self.chunk_num = (shot_len + chunk_len -1) // chunk_len
        else:
            # use k-anchors
            mask_step = self.stm_anchor_num - 1
            self.chunk_num = 1 + (max(0, len(self.mask_index) - self.stm_anchor_num) + (mask_step - 1)) // mask_step

    def getShotImageIndex(self, shot_index):
        # original frame index
        image_index_anchor = self.shot_list[shot_index]
        # assume it's consecutive
        image_index = image_index_anchor.reshape([-1,1]) + range(0, self.mask_index_factor_output[0], self.stm_step)
        # remove non-consecutive
        bad_row = np.where(image_index_anchor[1:] - image_index_anchor[:-1] != self.mask_index_factor_output[0])[0]
        image_index[bad_row, 1:] = -1
        image_index[-1, 1:] = -1
        # divide into consecutive chunks
        #image_index = image_index_anchor.reshape([-1,1]) + range(0, self.mask_index_factor_output[0], self.stm_step)
        # add the last frame
        self.image_index = np.unique(image_index[image_index >= 0])
                         
    def getShotMaskIndex(self, shot_index):
        if self.stm_mode == 'shot':
            self.mask_file_index = [shot_index]
            self.mask_index = [self.shot_list[shot_index][0]]
        elif self.stm_mode == 'index':
            index = self.shot_list[shot_index]
            # naive: assume the shot_list is continuous
            self.mask_file_index = np.where((self.mask_ids >= min(index)) * (self.mask_ids <= max(index)))[0]
            # remove redundant ones
            #self.mask_file_index = self.mask_file_index[np.in1d(self.mask_ids[self.mask_file_index], self.image_index)]
            self.mask_index = self.mask_ids[self.mask_file_index]
            
        self.mask_num = len(self.mask_index)
        # remove template frame
        self.image_index = removeArr(self.image_index, self.mask_index)

    def All_to_onehot(self, masks, K = 1):
        Ms = np.zeros((K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for k in range(K):
            Ms[k] = (masks == k).astype(np.uint8)
        return Ms

    def getChunkIndexLocal(self, shot_index, chunk_index):
        if self.stm_anchor_num == -1:
            chunk_len = self.stm_mem_len - self.mask_num 
            chunk_start = chunk_index * chunk_len
            chunk_len = min(chunk_len, len(self.image_index) - chunk_start)
        else:
            if self.chunk_num == 1:
                todo_index = range(len(self.image_index))
            elif chunk_index == 0:
                anchor_index = self.mask_index[self.stm_anchor_num-1]
                todo_index = np.where(self.image_index < anchor_index)[0]
            elif chunk_index == self.chunk_num - 1:
                anchor_index = self.mask_index[(self.stm_anchor_num - 1) * chunk_index]
                todo_index = np.where(self.image_index > anchor_index)[0]
            else:
                st_index = (self.stm_anchor_num - 1) * chunk_index
                anchor_index = self.mask_index[st_index : st_index + self.stm_anchor_num]
                todo_index = np.where((self.image_index > anchor_index[0])*(self.image_index < anchor_index[-1]))[0]

            chunk_len = len(todo_index)
            chunk_start = todo_index[0] if chunk_len>0 else -1 
        return chunk_start, chunk_len

    def loadSTMData(self, shot_index):
        N_masks = np.zeros([self.stm_mem_len, self.stm_height, self.stm_width], dtype=np.uint8)
        N_frames = np.zeros([self.stm_mem_len, self.stm_height, self.stm_width, 3], dtype=np.float32)
        # load initial template frames
        init_index = self.mask_index
        if self.stm_anchor_num != -1:
            init_index = init_index[:self.stm_anchor_num]
            self.mask_num = len(init_index)
            #import pdb; pdb.set_trace()
        else: # if bigger than the default size
            thres = 0.8
            if len(init_index) > thres * self.stm_mem_len:
                init_index = np.random.permutation(init_index)[: int(thres * self.stm_mem_len)] 

        for f,index in enumerate(init_index):
            mask_name = self.mask_files[self.mask_file_index[f]]
            if os.path.exists(mask_name):
                mask = Image.open(mask_name).convert('P')
                N_masks[f] = np.array(mask.resize((self.stm_width,self.stm_height),Image.NEAREST), dtype=np.uint8)

                image_file = self.image_template % index
                image = Image.open(image_file).convert('RGB')
                N_frames[f] = np.array(image.resize((self.stm_width,self.stm_height),Image.BILINEAR))/255.
                # print('init',mask_name,image_file)
        return N_frames, N_masks

    def updateSTMData(self, shot_index, chunk_index, N_frames, N_masks, mask_id_relabel, mask_id_relabel_inv):
        chunk_start, chunk_len = self.getChunkIndexLocal(shot_index, chunk_index)
        if chunk_len == 0:
            return None, None, 0
        else:
            # load mask
            if self.stm_mode == 'shot' and chunk_index > 0:
                # update mask
                # read from the previous prop mask
                mask_index = self.image_index[chunk_start - 1]
                mask_file = self.mask_template_output % self.convertIndexOutput(mask_index)
                if not os.path.exists(mask_file):
                    print('%s does not exist!' % mask_file)
                    # object disappears in the middle, use the first frame
                    mask_file = self.mask_files[shot_index]
                    #print('load first frame: %s!' % mask_file)
                    mask_index = self.image_index[0]
                mask = Image.open(mask_file).convert('P')
                N_masks[0] = np.array(mask.resize((self.stm_width,self.stm_height),Image.NEAREST), dtype=np.uint8)
                N_masks[0] = N_masks[0]
                K = N_masks[0].max()

                image_file = self.image_template % mask_index
                image = Image.open(image_file).convert('RGB')
                N_frames[0] = np.array(image.resize((self.stm_width,self.stm_height),Image.BILINEAR))/255.
            else:
                # index-based
                if self.stm_anchor_num != -1 and chunk_index > 0:
                    N_masks[:] = 0
                    N_frames[:] = 0
                    st_index = (self.stm_anchor_num - 1) * chunk_index
                    anchor_index = self.mask_file_index[range(st_index, min(len(self.mask_ids), st_index + self.stm_anchor_num))]
                    self.mask_num = len(anchor_index)
                    #import pdb; pdb.set_trace()
                    for i,j in enumerate(anchor_index):
                        mask_file = self.mask_files[j]
                        mask = Image.open(mask_file).convert('P')
                        N_masks[i] = np.array(mask.resize((self.stm_width,self.stm_height),Image.NEAREST), dtype=np.uint8)
                        image_file = self.image_template % self.mask_ids[j]
                        image = Image.open(image_file).convert('RGB')
                        N_frames[i] = np.array(image.resize((self.stm_width,self.stm_height),Image.BILINEAR))/255.
                        print(chunk_index, mask_file, image_file)
                K = N_masks[:self.mask_num].max()
            if K == 0:# no mask to propagate
                return None, None, 0

            # load images for propagation
            for f in range(chunk_len):
                index = self.image_index[chunk_start + f]
                image_file = self.image_template % index
                image = Image.open(image_file).convert('RGB')
                N_frames[self.mask_num + f] = np.array(image.resize((self.stm_width,self.stm_height),Image.BILINEAR))/255.
            
            # add the extra batch dimension
            if K >= len(mask_id_relabel):
                tmp = mask_id_relabel.copy()
                tmp_inv = mask_id_relabel_inv.copy()
                new_id = removeArr(np.unique(N_masks[:self.mask_num]), tmp)
                mask_id_relabel_inv = np.zeros(len(tmp_inv) + len(new_id), np.uint8)
                mask_id_relabel_inv[:len(tmp_inv)] = tmp_inv
                mask_id_relabel_inv[-len(new_id):] = new_id

                mask_id_relabel = np.zeros(K+1, np.uint8)
                mask_id_relabel[:len(tmp)] = tmp
                mask_id_relabel[new_id] = len(tmp_inv) + np.arange(len(new_id))


            num_objects = torch.LongTensor([mask_id_relabel.max()])
            Fs = torch.from_numpy(np.transpose(N_frames[:self.mask_num+chunk_len].copy(), (3, 0, 1, 2)).copy()[None,:]).float()
            Ms = torch.from_numpy(self.All_to_onehot(mask_id_relabel[N_masks[:self.mask_num+chunk_len]], num_objects+1).copy()[None,:]).float()
            return Fs, Ms, num_objects, mask_id_relabel, mask_id_relabel_inv

    def getShotOutputIndex(self, shot_index, chunk_index):
        # result_index: position in STM result array
        # output_index: position in original frame id
        chunk_start, chunk_len = self.getChunkIndexLocal(shot_index, chunk_index)
        result_index = np.arange(chunk_len)
        output_index = self.image_index[chunk_start + result_index]
        # remove already saved mask seg
        tosave_index = removeArr(self.shot_list[shot_index], self.mask_index)
        output_id = np.in1d(output_index, tosave_index)
        return self.mask_num + result_index[output_id], self.convertIndexOutput(output_index[output_id])


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
    print('save to '+ args.mask_template_output)
    dataloader = YouTopDataLoader(args)
    # util
    for shot_id in range(dataloader.shot_num):
        print('shot',shot_id)
        # load image index
        dataloader.getShotImageIndex(shot_id)
        # load mask index
        dataloader.getShotMaskIndex(shot_id)
        if dataloader.mask_num == 0:
            print('miss annotation')
            continue

        # copy or create template frames
        for i,index in enumerate(dataloader.mask_index): 
            mask_file_out = dataloader.mask_template_output % dataloader.convertIndexOutput(index)
            if args.redo or not os.path.exists(mask_file_out):
                mask_file_in = dataloader.mask_files[dataloader.mask_file_index[i]]
                if os.path.exists(mask_file_in):
                    shutil.copy(mask_file_in, mask_file_out)
                else:
                    output = Image.fromarray(np.zeros([dataloader.height, dataloader.width], np.uint8))
                    output.save(mask_file_out)

        anchor_num = dataloader.getShotAnchorLen(shot_id)
        if anchor_num == 1: # only one anchor frame, no need to propagate
            continue

        # preload the mask
        N_frames, N_masks = dataloader.loadSTMData(shot_id)
        # visualizeMask(N_frames[:dataloader.mask_num], N_masks[:dataloader.mask_num])
        # check volume
        mask_id_relabel, mask_id_relabel_inv = relabelMask(N_masks[:dataloader.mask_num])
        if mask_id_relabel.max() == 0:
            print('non-existent or empty mask files', [dataloader.mask_files[x] for x in dataloader.mask_file_index])
            continue

        dataloader.getShotChunkNum(shot_id)
        for chunk_id in range(dataloader.chunk_num):
            result_id, output_id = dataloader.getShotOutputIndex(shot_id, chunk_id)
            if args.redo or not os.path.exists(dataloader.mask_template_output % output_id[-1]):
                # mask_id_relabel can change
                Fs, Ms, num_objects, mask_id_relabel, mask_id_relabel_inv = dataloader.updateSTMData(shot_id, chunk_id, N_frames, N_masks, mask_id_relabel, mask_id_relabel_inv)
                if num_objects > 0:
                    print('chunk_id:',chunk_id, 'num_object:', num_objects)
                    #import pdb; pdb.set_trace()
                    pred, Es = Run_video(model, Fs, Ms, Fs.shape[2], num_objects,\
                                         Mem_every = args.stm_mem_step, Mem_number=None, st_frames=dataloader.mask_num)

                    pred = mask_id_relabel_inv[pred]
                    #import pdb; pdb.set_trace()
                    for z in range(len(result_id)):
                        if args.output_vis == 0:
                            output = Image.fromarray(pred[result_id[z]])
                        else:
                            pF = (Fs[0,:,result_id[z]].permute(1,2,0).numpy() * 255.).astype(np.uint8)
                            pE = pred[result_id[z]]
                            output = Image.fromarray(overlay_davis(pF, pE, colors))
                        output = output.resize((dataloader.width, dataloader.height),Image.NEAREST)
                        output.save(dataloader.mask_template_output % output_id[z])
