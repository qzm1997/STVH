import torch.nn as nn
import torch
import torch.nn.functional as F
from model.utils import *
from model.dcn.modules.deform_conv import *
class GraphAttentionLayer(nn.Module):
    def __init__(self,in_feature,out_feature,dropout,aplha,concat=True):
        super(GraphAttentionLayer,self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.dropout=dropout
        self.alpha=aplha
        self.concat=concat

        self.Wlinear=nn.Linear(in_feature,out_feature)
        # self.W=nn.Parameter(torch.empty(size=(batch_size,in_feature,out_feature)))
        nn.init.xavier_uniform_(self.Wlinear.weight,gain=1.414)

        self.aiLinear=nn.Linear(out_feature,1)
        self.ajLinear=nn.Linear(out_feature,1)
        # self.a=nn.Parameter(torch.empty(size=(batch_size,2*out_feature,1)))
        nn.init.xavier_uniform_(self.aiLinear.weight,gain=1.414)
        nn.init.xavier_uniform_(self.ajLinear.weight,gain=1.414)

        self.leakyRelu=nn.LeakyReLU(self.alpha)


    def getAttentionE(self,Wh):
        #重点改了这个函数
        Wh1=self.aiLinear(Wh)
        Wh2=self.ajLinear(Wh)
        Wh2=Wh2.view(Wh2.shape[0],Wh2.shape[2],Wh2.shape[1])
        # Wh1=torch.bmm(Wh,self.a[:,:self.out_feature,:])    #Wh:size(node,out_feature),a[:out_eature,:]:size(out_feature,1) => Wh1:size(node,1)
        # Wh2=torch.bmm(Wh,self.a[:,self.out_feature:,:])    #Wh:size(node,out_feature),a[out_eature:,:]:size(out_feature,1) => Wh2:size(node,1)

        e=Wh1+Wh2   #broadcast add, => e:size(node,node)
        return self.leakyRelu(e)

    def forward(self,h,adj, return_att=False):

        # print(h.device, self.Wlinear.weight.device)
        Wh=self.Wlinear(h)
        # Wh=torch.bmm(h,self.W)   #h:size(node,in_feature),W:size(in_feature,out_feature) => Wh:size(node,out_feature)
        e=self.getAttentionE(Wh)

        # zero_vec=-1e9*torch.ones_like(e)
        attention = adj * e
        # attention=torch.where(adj>0,e,zero_vec)
        attention=F.softmax(attention,dim=2)
        attention=F.dropout(attention,self.dropout,training=self.training)
        h_hat=torch.bmm(attention,Wh)  #attention:size(node,node),Wh:size(node,out_fature) => h_hat:size(node,out_feature)
        if return_att:
            return attention
        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

    def __repr__(self):
        return self.__class__.__name__+' ('+str(self.in_feature)+'->'+str(self.out_feature)+')'
class GAT(nn.Module):
    def __init__(self,in_feature,hidden_feature,out_feature,attention_layers,dropout,alpha):
        super(GAT,self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.hidden_feature=hidden_feature
        self.dropout=dropout
        self.alpha=alpha
        self.attention_layers=attention_layers

        self.attentions=nn.ModuleList([GraphAttentionLayer(in_feature,hidden_feature,dropout,alpha,True) 
                                       for i in range(attention_layers)])

        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

        self.out_attention=GraphAttentionLayer(attention_layers*hidden_feature,out_feature,dropout,alpha,False)


    def forward(self,h,adj):
        # print(h)
        h=F.dropout(h,self.dropout,training=self.training)

        h=torch.cat([attention(h,adj) for attention in self.attentions],dim=2)
        h=F.dropout(h,self.dropout,training=self.training)
        h=F.elu(self.out_attention(h,adj, True))
        return h
def build_tp_adj(bbox):
    # bbox = bbox.permute(1, 0, 2)
    iou_matrix = []
    for box in bbox:
        iou = jaccard(box, box)
        iou_matrix.append(iou)
    adj = torch.stack(iou_matrix)
    adj = torch.where(torch.isnan(adj), torch.full_like(adj, 1), adj)
    return adj
def build_sp_adj(bbox):
    T, N, _ = bbox.shape
    mean = torch.mean(bbox, dim=1)
    std = torch.std(bbox, dim=1)
    
    bbox = bbox.sub_(mean.unsqueeze(1)).div_(std.unsqueeze(1))
    adj = 1 - calc_pairwise_distance_3d(bbox, bbox)
    # adj = adj.mean(0).unsqueeze(0)
    adj = torch.where(torch.isnan(adj), torch.full_like(adj, 0), adj)
    return adj 
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
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.gat = GAT(dim, dim*2, dim, 3, 0.1, 0.2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, adj, dataset_name):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if dataset_name == 'CAD':
            mask = self.gat(x, adj).unsqueeze(1)
            mask = mask.repeat(1, self.num_heads, 1,1)
            mask = mask.softmax(dim=-1)
        else :
            mask = adj.unsqueeze(1).repeat(1, 8, 1, 1)
            mask = F.softmax(mask, dim=-1)
        # # attn = attn.softmax(dim=-1)
        attn = mask*attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, out_dim=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, graph=None, dataset_name = 'CAD'):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), graph, dataset_name))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



         
class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class ResBlock(nn.Module):
    def __init__(self, nf):
        super(ResBlock, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x 
class fusion(nn.Module):
    def __init__(self, dim):
        super(fusion, self).__init__()
        self.atten_s = nn.Sequential(
            nn.Linear(dim, dim//16,bias=False),
            nn.ELU(),
            nn.Linear(dim//16, 1, bias=False),
            nn.Sigmoid()
        )
        self.atten_t = nn.Sequential(
            nn.Linear(dim, dim//16, bias=False),
            nn.ELU(),
            nn.Linear(dim//16, 1, bias=False),
            nn.Sigmoid()
        )
        self.s_linear = nn.Linear(dim, dim)
        self.t_linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm([dim])
    def forward(self, s_feature, t_feature):
        s_atten = self.atten_s(s_feature)
        t_atten = self.atten_t(t_feature)
        t_feature = self.t_linear(t_atten * t_feature + t_feature)
        s_feature = self.s_linear(s_atten * s_feature + s_feature)
        out = t_feature.permute(1, 0, 2)+s_feature
        return self.norm(out)
class Hash_head(nn.Module):
    def __init__(self, in_dim, nbit, cls):
        super(Hash_head, self).__init__()
        self.hash_code = nn.Linear(in_dim, nbit, bias=False)
        self.tanh = nn.Tanh()
        self.cls_head = nn.Linear(nbit, cls)
    def forward(self, x):
        out1 = self.hash_code(x)
        out1 = self.tanh(out1)
        out2 = self.cls_head(out1)
        return out1, out2  
    

        
