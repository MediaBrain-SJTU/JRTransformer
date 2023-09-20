import sys
sys.path.append(".")
import os
import numpy as np
import torch
from dataset.dataset_3dpw import batch_denormalization, SoMoFDataset_3dpw_test
from model.model import JRTransformer

from torch.utils.data import DataLoader, SequentialSampler, DataLoader
from utils.config_3dpw import *
from utils.metrics import batch_MPJPE, batch_VIM
from utils.util import get_adj, get_connect

from datetime import datetime

class Tester:
    def __init__(self, args):
        # Set cuda device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(0)
        else:
            self.device = torch.device("cpu")
        print('Using device:', self.device)
        self.cuda_devices = args.device

        # Parameters
        self.batch_size = args.batch_size

        # Defining models
        self.model = JRTransformer(N=args.N, J=args.J, in_joint_size=args.in_joint_size, in_relation_size=args.in_relation_size, 
                                   feat_size=args.d_k, out_joint_size=args.out_joint_size, out_relation_size=args.out_relation_size,
                                   num_heads=args.num_heads, depth=args.depth).to(self.device)
        
        self.rc = args.rc
        dset_test = SoMoFDataset_3dpw_test(dset_path=somof_3dpw_test_data, seq_len=args.input_length+args.output_length, N=args.N, J=args.J)
        sampler_test = SequentialSampler(dset_test)
        self.test_loader = DataLoader(dset_test, sampler=sampler_test, batch_size=args.batch_size, num_workers=2, drop_last=False, pin_memory=True)
        
        edges = [(0, 1), (1, 8), (8, 7), (7, 0),
			 (0, 2), (2, 4),
			 (1, 3), (3, 5),
			 (7, 9), (9, 11),
			 (8, 10), (10, 12),
			 (6, 7), (6, 8)]
        self.adj = get_adj(args.N, args.J, edges)
        self.adj = self.adj.unsqueeze(0).unsqueeze(-1)
        self.conn = get_connect(args.N, args.J)
        self.conn = self.conn.unsqueeze(0).unsqueeze(-1)


        self.path = args.model_path
        
    def test(self):
        path = self.path
        checkpoint = torch.load(path)  
        self.model.load_state_dict(checkpoint['net']) 
        self.model.eval()

        all_mpjpe = np.zeros(5)
        all_vim = np.zeros(5)
        count = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                input_total_original, para = data
                input_total_original = input_total_original.float().cuda()
                input_total = input_total_original.clone()

                batch_size = input_total.shape[0]
                T=30
                input_total[..., [1, 2]] = input_total[..., [2, 1]]
                input_total[..., [4, 5]] = input_total[..., [5, 4]]

                if self.rc:
                    camera_vel = input_total[:, 1:30, :, :, 3:].mean(dim=(1, 2, 3)) # B, 3
                    input_total[..., 3:] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)

                input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 30, 6)
				# B, NxJ, T, 6

                input_joint = input_total[:,:, :16]
				
                pos = input_total[:,:,:16,:3]
                pos_i = pos.unsqueeze(-3)
                pos_j = pos.unsqueeze(-4)
                pos_rel = pos_i - pos_j
                dis = torch.pow(pos_rel, 2).sum(-1)
                dis = torch.sqrt(dis)
                exp_dis = torch.exp(-dis)
                input_relation = torch.cat((exp_dis, self.adj.repeat(batch_size, 1, 1, 1), self.conn.repeat(batch_size, 1, 1, 1)), dim=-1)

                pred_vel = self.model.predict(input_joint, input_relation)
                pred_vel = pred_vel[:, :, 16:]

				
                pred_vel = pred_vel.permute(0, 2, 1, 3)

                if self.rc:
                    pred_vel = pred_vel + camera_vel[:, None, None]

				# B, T, NxJ, 3
                pred_vel[..., [1, 2]] = pred_vel[..., [2, 1]]
				# Cumsum velocity to position with initial pose.
                motion_gt = input_total_original[...,:3].view(batch_size, T, -1, 3)
                motion_pred = (pred_vel.cumsum(dim=1) + motion_gt[:, 15:16])
				
				# Apply denormalization.
                motion_pred = batch_denormalization(motion_pred.cpu(), para).numpy()               
                motion_gt = batch_denormalization(motion_gt.cpu(), para).numpy() 

                metric_MPJPE = batch_MPJPE(motion_gt[:, 16:, :13, :], motion_pred[:, :, :13, :])
                all_mpjpe += metric_MPJPE

                metric_VIM = batch_VIM(motion_gt[:, 16:, :13, :], motion_pred[:, :, :13, :])
                all_vim += metric_VIM
                
                count += batch_size

            all_mpjpe *= 100
            all_vim *= 100
            all_mpjpe /= count
            all_vim /= count
            print('Test MPJPE:\t avg: {:.2f} | 100ms: {:.2f} | 240ms: {:.2f} | 500ms: {:.2f} | 640ms: {:.2f} | 900ms: {:.2f}'.format(all_mpjpe.mean(), all_mpjpe[0],  all_mpjpe[1],  all_mpjpe[2],  all_mpjpe[3],  all_mpjpe[4]))
            print('Test VIM:\t avg: {:.2f} | 100ms: {:.2f} | 240ms: {:.2f} | 500ms: {:.2f} | 640ms: {:.2f} | 900ms: {:.2f}'.format(all_vim.mean(), all_vim[0],  all_vim[1],  all_vim[2],  all_vim[3],  all_vim[4]))    
        return all_vim.mean()

if __name__=='__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    tester = Tester(args)
    tester.test()