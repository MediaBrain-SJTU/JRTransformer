import sys
sys.path.append(".")
import os
import random
import numpy as np
import torch
import time
from dataset.dataset_3dpw import SoMoFDataset_3dpw, batch_denormalization, SoMoFDataset_3dpw_test
from model.model import JRTransformer
from torch.utils.data import DataLoader, SequentialSampler, DataLoader
from utils.metrics import batch_MPJPE, batch_VIM
from utils.config_3dpw import *
from utils.util import rotate_Y, get_adj, get_connect, distance_loss, relation_loss, process_pred
from datetime import datetime

class Trainer:
    def __init__(self, args):
        # Set cuda device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(0)
        else:
            self.device = torch.device("cpu")
        print('Using device:', self.device)
        self.cuda_devices = args.device

        # Training parameters
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.num_epoch = args.num_epoch
        self.weight_loss_pred = args.weight_loss_pred
        self.weight_loss_recon = args.weight_loss_recon 
        self.weight_loss_aux = args.weight_loss_aux

        # Defining models
        self.model = JRTransformer(N=args.N, J=args.J, in_joint_size=args.in_joint_size, in_relation_size=args.in_relation_size, 
                                   feat_size=args.d_k, out_joint_size=args.out_joint_size, out_relation_size=args.out_relation_size,
                                   num_heads=args.num_heads, depth=args.depth).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=args.step_size, gamma=args.gamma)
        
        self.rc = args.rc
        dset_train = SoMoFDataset_3dpw(dset_path=preprocess_3dpw_train_data, seq_len=args.input_length+args.output_length, N=args.N, J=args.J, split_name="train")
        sampler_train = SequentialSampler(dset_train)
        self.train_loader = DataLoader(dset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=2, drop_last=False, pin_memory=True)
        print("Load Train set!")

        dset_val = SoMoFDataset_3dpw(dset_path=preprocess_3dpw_train_data, seq_len=args.input_length+args.output_length, N=args.N, J=args.J, split_name="valid")
        sampler_val = SequentialSampler(dset_val)
        self.val_loader = DataLoader(dset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=2, drop_last=False, pin_memory=True)   
        print("Load Valid set!")

        dset_test = SoMoFDataset_3dpw_test(dset_path=somof_3dpw_test_data, seq_len=args.input_length+args.output_length, N=args.N, J=args.J)
        sampler_test = SequentialSampler(dset_test)
        self.test_loader = DataLoader(dset_test, sampler=sampler_test, batch_size=args.batch_size, num_workers=2, drop_last=False, pin_memory=True)
        print("Load Test set!")
        
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

        self.pretrain_path = args.pretrain_path

        # Training log
        self.log_dir = args.log_dir
        self.model_dir = 'h_' + str(args.d_k) + '_' + \
                'd_' + str(args.depth) + '_' + \
                'nh_' + str(args.num_heads) + '_' +\
                datetime.now().strftime('%m%d%H%M') + '/'
        if not os.path.exists(self.log_dir + self.model_dir):
            os.makedirs(self.log_dir + self.model_dir)

    def test(self):
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
            with open(os.path.join(self.log_dir + self.model_dir, 'log.txt'), 'a+') as log:
                log.write('Test MPJPE:\t avg: {:.2f} | 100ms: {:.2f} | 240ms: {:.2f} | 500ms: {:.2f} | 640ms: {:.2f} | 900ms: {:.2f}\n'.format(all_mpjpe.mean(), all_mpjpe[0],  all_mpjpe[1],  all_mpjpe[2],  all_mpjpe[3],  all_mpjpe[4]))
                log.write('Test VIM:\t avg: {:.2f} | 100ms: {:.2f} | 240ms: {:.2f} | 500ms: {:.2f} | 640ms: {:.2f} | 900ms: {:.2f}\n'.format(all_vim.mean(), all_vim[0],  all_vim[1],  all_vim[2],  all_vim[3],  all_vim[4]))    
        return all_vim.mean()
    
    def train(self):
        start_time = time.time()
        steps = 0
        losses = []
        start_epoch = 0
        self.best_eval=100

        if self.pretrain_path != '':
            checkpoint = torch.load(self.pretrain_path)  
            self.model.load_state_dict(checkpoint['net']) 

        for train_iter in range(start_epoch, self.num_epoch):
            print("Epoch:", train_iter)
            print("Time since start:", (time.time() - start_time) / 60.0, "minutes.")
            self.model.train()
            self.epoch = train_iter
            all_mpjpe = np.zeros(5)
            all_vim = np.zeros(5)

            count = 0
            for i, data in enumerate(self.train_loader):
                input_total, _ = data
                input_total = input_total.float().cuda()
                input_total[..., [1, 2]] = input_total[..., [2, 1]]
                input_total[..., [4, 5]] = input_total[..., [5, 4]]


                # input_total: normalized poses with the shape [batch_size, T_p+T_f, N, J, D],
                # 		where T_p=16 stands for past frames, 
                # 			  T_f=14 stands for future frames,
                # 			  N=2 stands for interacting agents, 
                # 			  J=13 stands for joints number, 
                # 			  D=6 stands for input feature dimension, 3 for normalized position, 3 for velocity.

                batch_size = input_total.shape[0]

                if self.rc:
                    camera_vel = input_total[:, 1:30, :, :, 3:].mean(dim=(1, 2, 3)) # B, 3
                    input_total[..., 3:] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)

                input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 30, 6)
                # B, NxJ, T, 6
                angle = random.random()*360
                # random rotation
                input_total = rotate_Y(input_total, angle)
                input_total *= (random.random()*0.4+0.8)

                input_joint = input_total[:,:, :16]

                pos = input_total[:,:,:,:3]
                pos_i = pos.unsqueeze(-3)
                pos_j = pos.unsqueeze(-4)
                pos_rel = pos_i - pos_j
                dis = torch.pow(pos_rel, 2).sum(-1)
                dis = torch.sqrt(dis)
                exp_dis = torch.exp(-dis)

                exp_dis_in = exp_dis[:, :, :,:16]
                dis_recon = exp_dis[:,:,:,:16]
                dis_pred = exp_dis[:,:,:,16:]
                input_relation = torch.cat((exp_dis_in, self.adj.repeat(batch_size, 1, 1, 1), self.conn.repeat(batch_size, 1, 1, 1)), dim=-1)
                # B, NxJ, T_p, 6

                pred_vel, pred_relation, pred_vel_aux, pred_relation_aux = self.model(input_joint, input_relation)
                recon_vel, pred_vel, recon_vel_aux, pred_vel_aux = process_pred(pred_vel, pred_vel_aux)

                gt_vel = input_total[..., 3:]
                # [B, NxJ, T=30, 3]
                gt_vel_x = gt_vel[:, :, :16]
                gt_vel_y = gt_vel[:, :, 16:]

                loss_recon = distance_loss(recon_vel, gt_vel_x)
                loss_pred = distance_loss(pred_vel, gt_vel_y)
                loss_relation_recon= relation_loss(pred_relation[..., :16], dis_recon)
                loss_relation_pred = relation_loss(pred_relation[..., 16:], dis_pred)
                loss_aux_recon, loss_aux_pred, loss_aux_relation_recon, loss_aux_relation_pred = 0, 0, 0, 0
                for i_ in range(len(recon_vel_aux)):
                    loss_aux_recon = loss_aux_recon + distance_loss(recon_vel_aux[i_], gt_vel_x)
                    loss_aux_pred  = loss_aux_pred  + distance_loss(pred_vel_aux[i_], gt_vel_y)
                    loss_aux_relation_recon  = loss_aux_relation_recon + relation_loss(pred_relation_aux[i_][..., :16], dis_recon)
                    loss_aux_relation_pred  = loss_aux_relation_pred + relation_loss(pred_relation_aux[i_][..., 16:], dis_pred)

                loss =  loss_pred * self.weight_loss_pred + loss_recon * self.weight_loss_recon + loss_relation_pred * self.weight_loss_pred + loss_relation_recon * self.weight_loss_recon + (loss_aux_recon + loss_aux_pred + loss_aux_relation_pred + loss_aux_relation_recon) * self.weight_loss_aux

                count += batch_size
                
                self.opt.zero_grad()
                # Backward pass: compute gradient of the loss with respect to parameters.
                loss.backward()
                # Perform gradient clipping.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                # Calling the step function to update the parameters.
                self.opt.step()

                losses.append([loss.item()])
               
                steps += 1
            self.scheduler_model.step()

            print("Loss", np.array(losses).mean())

            all_mpjpe *= 100
            all_vim *= 100
            all_mpjpe /= count
            all_vim /= count
            with open(os.path.join(self.log_dir + self.model_dir, 'log.txt'), 'a+') as log:
                log.write('Epoch: {}, Train Loss: {},\n'.format(train_iter, np.array(losses).mean()))
               
            eval = self.test()

            if eval < self.best_eval:
                self.best_eval = eval
                self.best_model = self.model.state_dict()
                print('best_eval:{}'.format(self.best_eval))
                checkpoint = {
                        "net": self.model.state_dict(),
                    }
                torch.save(checkpoint, self.expt_dir + self.model_dir + 'best.pt')    
            
            if train_iter % 10 == 9:
                checkpoint = {
                    "net": self.model.state_dict(),
                }
                torch.save(checkpoint, self.expt_dir + self.model_dir + 'epoch_{}.pt'.format(train_iter))



if __name__=='__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    trainer = Trainer(args)
    trainer.train()