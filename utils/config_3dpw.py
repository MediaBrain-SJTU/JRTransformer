import argparse
import os

## path
original_3dpw_dir = 'data/3DPW/'
preprocess_3dpw_train_data = 'data/poseData.pkl'
somof_3dpw_test_data = 'data/test_in.json'
AMASS_dir = '/DB/public/AMASS/'
skel_path = 'data/smpl_skeleton.npz'

## data
dataset_skip = 2
strike = 2
input_length = 16
output_length = 14
num_joint = 13
num_person = 2

## train & test
# pretrain 
pretrain_batch_size = 256
pretrain_learning_rate = 1e-3
pretrain_num_epoch = 100
pretrain_step_size = 10
pretrain_gamma = 0.8
#finetune
batch_size = 128
learning_rate = 1e-4
num_epoch = 100
step_size = 10
gamma = 0.8

cuda_devices = '0'
pretrain_path = ''
model_path = 'output/best_3dpw.pt'
weight_loss_pred = 10.0
weight_loss_recon = 1.0
weight_loss_aux = 1.0

## model setting
depth = 4
num_heads = 8
d_k = 128
in_joint_size = 16 * 6
in_relation_size = 18
out_joint_size = 30 * 3
out_relation_size = 30

def parse_args():
    parser = argparse.ArgumentParser()

    ## path setting
    parser.add_argument('--log_dir', 
                        type=str, 
                        default=os.path.join(os.getcwd(), 'logs/'),
                        help='dir for saving logs')

    ## train & test setting
    parser.add_argument('--pretrain_batch_size', 
                        type=int, 
                        default=pretrain_batch_size,
                        help='batch size to pretrain on AMASS')
    parser.add_argument('--pretrain_lr', 
                        type=float, 
                        default=pretrain_learning_rate,
                        help='initial learing rate for pretrain')
    parser.add_argument('--pretrain_num_epoch', 
                        type=int, 
                        default=pretrain_num_epoch,
                        help='#epochs to pretrain')
    parser.add_argument('--pretrain_step_size', 
                        type=int, 
                        default=pretrain_step_size,
                        help='learning rate decay step')
    parser.add_argument('--pretrain_gamma', 
                        type=float, 
                        default=pretrain_gamma,
                        help='learning rate decay rate')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=batch_size,
                        help='batch size to train')
    parser.add_argument('--lr', 
                        type=float, 
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=num_epoch,
                        help='#epochs to train')
    parser.add_argument('--step_size', 
                        type=int, 
                        default=step_size,
                        help='learning rate decay step')
    parser.add_argument('--gamma', 
                        type=float, 
                        default=gamma,
                        help='learning rate decay rate')
    
    parser.add_argument('--device', 
                        type=str, 
                        default=cuda_devices,
                        help='set device for training')
    parser.add_argument('--pretrain_path',
                        type=str,
                        default=pretrain_path,
                        help='path of the model pretrained on AMASS')
    parser.add_argument('--model_path',
                        type=str,
                        default=model_path,
                        help='path of the model finetuned on 3dpw')
    
    parser.add_argument('--weight_loss_pred',
                        type=float,
                        default=weight_loss_pred,
                        help='loss weight of predicted pose loss')
    parser.add_argument('--weight_loss_recon',
                        type=float,
                        default=weight_loss_recon,
                        help='loss weight of reconstruction pose loss')
    parser.add_argument('--weight_loss_aux',
                        type=float,
                        default=weight_loss_aux,
                        help='loss weight of deep supervision')
   
    ## model setting
    parser.add_argument('--depth',
                        type=int,
                        default=depth,
                        help='model depth')
    parser.add_argument('--num_heads',
                        type=int,
                        default=num_heads,
                        help='num heads of multihead attention')
    parser.add_argument('--d_k',
                        type=int,
                        default=d_k,
                        help='feature dimension')
    parser.add_argument('--in_joint_size',
                        type=int,
                        default=in_joint_size,
                        help='input joint dimension')
    parser.add_argument('--in_relation_size',
                        type=int,
                        default=in_relation_size,
                        help='input relation dimension')
    parser.add_argument('--out_joint_size',
                        type=int,
                        default=out_joint_size,
                        help='output joint dimension')
    parser.add_argument('--out_relation_size',
                        type=int,
                        default=out_relation_size,
                        help='output relation dimension')
   
    ## data process setting
    parser.add_argument('--skip', 
                        type=int, 
                        default=dataset_skip,
                        help='down sample rate')
    parser.add_argument('--strike', 
                        type=int, 
                        default=strike,
                        help='number of frames that we have to skip')
    parser.add_argument('--rc',
                        type=bool,
                        default=True,
                        help='whether to remove camera movement')
    parser.add_argument('--input_length', 
                        type=int, 
                        default=input_length,
                        help='input sequence length')
    parser.add_argument('--output_length', 
                        type=int, 
                        default=output_length,
                        help='output sequence length')
    parser.add_argument('--J', 
                        type=int, 
                        default=num_joint,
                        help='number of joints')    
    parser.add_argument('--N', 
                        type=int, 
                        default=num_person,
                        help='number of person in a scene')    
    
    args = parser.parse_args()

    return args
