import numpy as np
import torch
import math

def rotate_Y(input, beta):
	'''
	beta: angle, range 0~360.
	'''
	output = torch.zeros_like(input)
	beta = beta * (math.pi / 180) # angle to radian
	output[..., 0] = np.cos(beta)*input[..., 0] + np.sin(beta)*input[..., 2]
	output[..., 2] = -np.sin(beta)*input[..., 0] + np.cos(beta)*input[..., 2]
	output[..., 1] = input[..., 1]

	output[..., 3] = np.cos(beta)*input[..., 3] + np.sin(beta)*input[..., 5]
	output[..., 5] = -np.sin(beta)*input[..., 3] + np.cos(beta)*input[..., 5]
	output[..., 4] = input[..., 4]
	return output

def get_adj(N, J, edges):
	adj = np.eye(N*J)
	for edge in edges:
		for i in range(N):
			adj[edge[0] + i * J, edge[1] + i * J] = 1
			adj[edge[1] + i * J, edge[0] + i * J] = 1
	return torch.from_numpy(adj).float().cuda()

def get_connect(N, J):
	conn = np.zeros((N*J, N*J))
	conn[:J, :J] = 1
	conn[J:, J:] = 1
	return torch.from_numpy(conn).float().cuda()

def distance_loss(target, pred):
    mse_loss = (pred - target) ** 2
    mse_loss = mse_loss.sum(-1)
    mse_loss = mse_loss.sqrt()
    loss = mse_loss.mean()
    return loss

def relation_loss(target, pred):
    mse_loss = torch.abs(pred - target) 
    loss = mse_loss.mean()
    return loss

def process_pred(pred_vel, pred_vel_aux):
	pred_vel_x = pred_vel[:, :, :16]
	pred_vel_y = pred_vel[:, :, 16:]
	pred_vel_aux_x = []
	pred_vel_aux_y = []
	for pred_ in pred_vel_aux:
		pred_vel_aux_x.append(pred_[:, :, :16])
		pred_vel_aux_y.append(pred_[:, :, 16:])
	return pred_vel_x, pred_vel_y, pred_vel_aux_x, pred_vel_aux_y