import torch
import torch.nn as nn
from functools import partial

class PositionalEmbedding(nn.Module):
	def __init__(self, N, J, embed_size, dropout=0.1):
		super().__init__()
		self.N = N
		self.J = J
		self.joint = nn.Parameter(torch.zeros(J, embed_size))
		self.person = nn.Parameter(torch.zeros(N, embed_size))
		self.dropout = nn.Dropout(p=dropout)
		torch.nn.init.normal_(self.joint, std=.02)
		torch.nn.init.normal_(self.person, std=.02)

	def forward_spatial(self):
		p_person = self.person.repeat_interleave(self.J, dim=0)
		p_joint = self.joint.repeat(self.N, 1)
		p = p_person + p_joint
		return self.dropout(p)
	
	def forward_relation(self):
		p = self.forward_spatial()
		p_i = p.unsqueeze(-2)
		p_j = p.unsqueeze(-3)
		p = p_i + p_j
		return self.dropout(p)	


class MLP(nn.Module):
	def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)
		return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.J_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.R_conv = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.R_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, joint_feature, relation_feature, mask=None):
        B, N, C = joint_feature.shape
        H = self.num_heads
        HS = C // self.num_heads
        J_qkv = self.J_qkv(joint_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4) #[3, B, #heads, N, C//#heads]
        R_qkv = self.R_qk(relation_feature).reshape(B, N, N, 2, H, HS).permute(3, 0, 4, 1, 2, 5)  #[3, B, #heads, N, N, C//#heads]

        J_q, J_k, J_v = J_qkv[0], J_qkv[1], J_qkv[2]   #[B, #heads, N, C//#heads]
        R_q, R_k = R_qkv[0], R_qkv[1]  #[B, #heads, N, N, C//#heads]

        attn_J = (J_q @ J_k.transpose(-2, -1)) # [B, #heads, N, N]
        attn_R_linear = self.R_conv(relation_feature).reshape(B, N, N, H).permute(0, 3, 1, 2)  #[B, #heads, N, N]
        attn_R_qurt = (R_q.unsqueeze(-2) @ R_k.unsqueeze(-1)).squeeze() # [B, #heads, N, N]

        attn = (attn_J + attn_R_linear + attn_R_qurt) * self.scale #[B, #heads, N, N]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # [B, #heads, N, N]

        x = (attn @ J_v).transpose(1, 2).reshape(B, N, C) #[B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                
		self.norm_attn1 = norm_layer(dim)
		self.norm_attn2 = norm_layer(dim)
		self.norm_joint = norm_layer(dim)
                
		self.norm_relation1 = norm_layer(dim*4)
		self.norm_relation2 = norm_layer(dim)
        
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                
		self.mlp_joint = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
                
		self.mlp_relation1 = Mlp(in_features=dim*4, hidden_features=dim*4, out_features=dim, act_layer=act_layer, drop=drop)
		self.mlp_relation2 = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
	
	def forward(self, joint_feature, relation_feature, mask=None):
		B, N, C = joint_feature.shape
        ## joint feature update through attention mechanism
		joint_feature = joint_feature + self.drop_path(self.attn(self.norm_attn1(joint_feature), self.norm_attn2(relation_feature), mask))
		joint_feature = joint_feature + self.drop_path(self.mlp_joint(self.norm_joint(joint_feature)))
		
        ## relation feature update 
        # M_ij = cat(J_i, J_j, R_ij, R_ji)
		joint_i = joint_feature.unsqueeze(1).repeat(1, N, 1, 1) #[B, N, N, D] J_i
		joint_j = joint_feature.unsqueeze(2).repeat(1, 1, N, 1) #[B, N, N, D] J_j
		relation_rev = relation_feature.swapaxes(-2, -3) #[B, N, N, D] E_ji
		relation_input = torch.cat((relation_feature, relation_rev, joint_i, joint_j), -1)
                
        # U_ij = R_ij + MLP(Norm(M_ij)), R_ij' = U_ij + MLP(Norm(U_ij))
		relation_feature = relation_feature + self.drop_path(self.mlp_relation1(self.norm_relation1(relation_input)))
		relation_feature = relation_feature + self.drop_path(self.mlp_relation2(self.norm_relation2(relation_feature)))
		return joint_feature, relation_feature


class JRTransformer(nn.Module):
	def __init__(self, N=2, J=13, in_joint_size=16*6, in_relation_size=18, feat_size=128, out_joint_size=30*3, out_relation_size=30, num_heads=8, depth=4, norm_layer=nn.LayerNorm):
		super().__init__()
		
		self.joint_encoder = MLP(in_joint_size, feat_size, (256, 256))
		self.relation_encoder = MLP(in_relation_size, feat_size, (256, 256))
		self.pe = PositionalEmbedding(N, J, feat_size)
		self.norm_layer = norm_layer(feat_size)

		self.attn_encoder = nn.ModuleList([
			Block(feat_size, num_heads, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
			for i in range(depth)])
		
		self.joint_decoder = MLP(feat_size, out_joint_size)
		self.relation_decoder = MLP(feat_size, out_relation_size)

		self.initialize_weights()
 
	def initialize_weights(self):
		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		
		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)


	def forward(self, x_joint, x_relation):
		B, NJ, T, D = x_joint.shape
		x_joint = x_joint.view(B, NJ, -1)
		x_relation = x_relation.view(B, NJ, NJ, -1)

		feat_x_joint = self.joint_encoder(x_joint)
		feat_x_relation = self.relation_encoder(x_relation)
		# B, NJ, 128

		pe_joint = self.pe.forward_spatial()
		pe_relation = self.pe.forward_relation()
    
		pred_aux_joint = [self.joint_decoder(feat_x_joint).contiguous().view(B, NJ, -1, 3)]
		pred_aux_relation = [self.relation_decoder(feat_x_relation).contiguous().view(B, NJ, NJ, -1)]
		
		for i in range(len(self.attn_encoder)):
			blk = self.attn_encoder[i]
			feat_x_joint = feat_x_joint + pe_joint
			feat_x_relation = feat_x_relation + pe_relation
			feat_x_joint, feat_x_relation = blk(feat_x_joint, feat_x_relation)
			pred_aux_joint.append(self.joint_decoder(feat_x_joint).contiguous().view(B, NJ, -1, 3))
			pred_aux_relation.append(self.relation_decoder(feat_x_relation).contiguous().view(B, NJ, NJ, -1))
		pred = self.joint_decoder(feat_x_joint).contiguous().view(B, NJ, -1, 3)
		pred_relation = self.relation_decoder(feat_x_relation).contiguous().view(B, NJ, NJ, -1)
		
		return pred, pred_relation, pred_aux_joint, pred_aux_relation

	def predict(self, x_joint, x_relation):
		B, NJ, T, D = x_joint.shape
		x_joint = x_joint.view(B, NJ, -1)
		x_relation = x_relation.view(B, NJ, NJ, -1)

		feat_x_joint = self.joint_encoder(x_joint)
		feat_x_relation = self.relation_encoder(x_relation)
		# B, NJ, 128

		pe_joint = self.pe.forward_spatial()
		pe_relation = self.pe.forward_relation()
		
		for i in range(len(self.attn_encoder)):
			blk = self.attn_encoder[i]
			feat_x_joint = feat_x_joint + pe_joint
			feat_x_relation = feat_x_relation + pe_relation
			feat_x_joint, feat_x_relation = blk(feat_x_joint, feat_x_relation)
		pred = self.joint_decoder(feat_x_joint).contiguous().view(B, NJ, -1, 3)
		
		return pred


