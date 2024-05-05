from model.backbone.backbone import *
from model.utils import *
from roi_align.roi_align import RoIAlign  
from model.transformer import *
from torch.nn.parameter import Parameter
from model.self_attention import *
import collections
def build_tp_adj(bbox):
    # bbox = bbox.permute(1, 0, 2)
    iou_matrix = []
    for box in bbox:
        iou = jaccard(box, box)
        iou_matrix.append(iou)
    adj = torch.stack(iou_matrix)
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
class GraphConvolution(nn.Module):                            
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):             # 这里代码做了简化如 3.2节。
        support = torch.mm(input, self.weight) # (2708, 16) = (2708, 1433) X (1433, 16)
        output = torch.spmm(adj, support)      # (2708, 16) = (2708, 2708) X (2708, 16)
        if self.bias is not None:
            return output + self.bias          # 加上偏置 (2708, 16)
        else:
            return output                      # (2708, 16)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):                                             # 定义两层GCN
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
class STVH_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(STVH_volleyball, self).__init__()
        self.cfg=cfg
        
        T, N=self.cfg.num_frames, self.cfg.num_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG=self.cfg.num_graph
        
        self.W_T=nn.Parameter(torch.ones(12, 10, 10))  # [1, G, 1, 1]
        self.W_S=nn.Parameter(torch.ones(10, 12, 12))  # [1, G, 1, 1]
        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained = True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained = True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained = True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.dcn = ResBlock_3d(nf=512)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        # self.fc_emb_1 = nn.Linear(K*K*D,NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        self.conv_1 = nn.Conv2d(512, NFB, (5, 5), stride=(5, 5))
        self.nl_pos_1 = nn.LayerNorm([NFB])
        self.gcn = GCN(9, NFB, 128, 0.1)
        #self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')
        dpr = [x.item() for x in torch.linspace(0, 0.1, 3)] 
        self.Aggregation1 = self_att_blk(NFB, 3, 8, NFB//8, NFB, 0.1)
        self.Aggregation2 = self_att_blk(NFB, 3, 8, NFB//8, NFB, 0.1)
        self.fusion = fusion(NFB)
        self.fc_actions = nn.Linear(NFB, 9)
        self.dpi_nl = nn.LayerNorm([12,in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.boxe2embed = nn.Linear(4, NFB)
        # self.activities_embd = nn.Linear(in_dim, 16)
        self.head = Hash_head(in_dim=in_dim, nbit=128, cls=self.cfg.num_activities)
        self.sp_out = nn.Linear(12*12,128)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath, map_location='cpu')
        self.backbone.load_state_dict(state['backbone_state_dict'])
        # self.fc_emb_1.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k,v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num +=1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num)+' parameters loaded for '+prefix)


    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4]==torch.Size([OH,OW])
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        # boxes_in_flat.requires_grad=False
        # boxes_idx_flat.requires_grad=False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features=boxes_features.reshape(B,T,N,-1)  #B,T,N, D*K*K
        boxes_features = boxes_features.reshape(T, N,-1, 5, 5)
        boxes_features = boxes_features.permute(1, 2, 0, 3, 4).contiguous()
        boxes_features = self.dcn(boxes_features).permute(2, 0, 1, 3, 4).contiguous()
        
        boxes_features = self.conv_1(boxes_features.reshape(T*N, -1, 5, 5)).reshape(B, T, N, -1)
        boxes_features = self.nl_emb_1(boxes_features)
     
        boxes_features=F.relu(boxes_features, inplace = True)[0]
        # features = boxes_features 
        tP_features = boxes_features.permute(1, 0, 2)
        tp_adj = build_tp_adj(boxes_in[0].permute(1, 0, 2))
        tp_adj = tp_adj*self.W_T

        tP_features = self.Aggregation1(tP_features, tp_adj)
        sp_feature = tP_features.permute(1,0,2)
        sp_adj = build_sp_adj(torch.clone(boxes_in[0]))
        sp_adj = sp_adj * self.W_S
        sp_adj = torch.mean(sp_adj, dim=0).unsqueeze(0)
        sp_feature = torch.mean(sp_feature, dim=0).unsqueeze(0)
        sp_feature = self.Aggregation2(sp_feature, sp_adj)
        action_feature = torch.mean(tP_features, dim=1)
        action_feature = self.nl_emb_1(action_feature)

        boxes_states_pooled_flat = sp_feature.reshape(1,B*N, -1)
        boxes_states_pooled_flat = self.dpi_nl(boxes_states_pooled_flat)
        boxes_states_pooled_flat = F.elu(boxes_states_pooled_flat, inplace = True)
        boxes_states_pooled_flat = self.dropout_global(boxes_states_pooled_flat)

        hash_out, activities_scores = self.head(boxes_states_pooled_flat)  #B*T, acty_num
        action_scores = self.fc_actions(action_feature) 
        activities_scores = activities_scores.reshape(B, N, -1)
        activities_scores = torch.mean(activities_scores,dim=1).reshape(B,-1)
        hash_out = hash_out.reshape(B, N, -1)
        hash_out = torch.mean(hash_out,dim=1)
        sp_ = self.gcn(action_scores, sp_adj[0])
        sp_ = torch.mean(sp_, dim=0).unsqueeze(0)
        return {'activities':activities_scores,
                "actions":action_scores,
                'hashcode':hash_out,
                'sp_adj':sp_} # actions_scores, activities_scores
class STVH_collective(nn.Module):
    def __init__(self, cfg):
        super(STVH_collective, self).__init__()
        self.cfg = cfg
        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.dcn = ResBlock_3d(nf=512)
        self.conv_1 = nn.Conv2d(512, NFB, (5, 5), stride=(5, 5))
        self.nl_emb_1 = nn.LayerNorm([NFB])
        self.W_T=nn.Parameter(torch.ones(13, 10, 10))  # [1, G, 1, 1]
        self.W_S=nn.Parameter(torch.ones(10, 13, 13))
        self.box2embed = nn.Linear(4, NFB)
        in_dim = NFB
        print_log(cfg.log_path, 'Deactivate lite model inference.')
        dpr = [x.item() for x in torch.linspace(0, 0.1, 5)] 
        self.Aggregation1 = nn.ModuleList([
            Block(dim=1024, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=True,
                drop=dpr[i], attn_drop=0., drop_path=0., init_values=0.,out_dim=10) for i in range(4)
        ])
        self.Aggregation2 = nn.ModuleList([
            Block(dim=1024, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=True,
                drop=dpr[i], attn_drop=0., drop_path=0.,init_values=0.,out_dim=12) for i in range(4)]
        )
        # self.T_gat = GAT(1024, 1024*2, 1024, 3, 0.1, 0.2)
        # self.S_gat = GAT(1024, 1024*2, 1024, 3, 0.1, 0.2)
        self.gcn = GCN(self.cfg.num_actions, NFB, 128, 0.1)
        self.fusion = fusion(NFB)
        self.dpi_nl = nn.LayerNorm([in_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.fc_actions = nn.Linear(in_dim, self.cfg.num_actions)
        # Lite Dynamic inference
        self.head = Hash_head(in_dim, 128, self.cfg.num_activities)
        # self.sp_out = nn.Linear(12*12, 128)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        postion = torch.clone(boxes_in)
        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        # features = self.box2embed(boxes_in)
        # boxes_features_all = features.reshape(B, T, MAX_N, NFB)
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, 512, 5, 5)  # B*T,MAX_N, D*K*K

        

        actions_scores = []
        activities_scores = []
        hash_code = []
        sp_feat = []
        sp_adj = []
        tp_adj = []
        # for p in postion:
        #     sp = build_sp_adj(p) 
        #     tp = build_tp_adj(p.permute(1,0,2)) 
        #     sp_adj.append(sp)
        #     tp_adj.append(tp)
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :,:,:]# 1,T,N,NFB
            boxes_features = boxes_features.permute(1,2,0,3,4)
            boxes_features = self.dcn(boxes_features.contiguous()).permute(2,0,1,3,4)
            boxes_features = self.conv_1(boxes_features.reshape(T*N, -1, 5, 5)).reshape(T, N, -1)
            boxes_features = self.nl_emb_1(boxes_features)
            boxes_features = F.relu(boxes_features)
            
            p = postion[b,:,:N,:]
            sp_adj = build_sp_adj(torch.clone(p))
            tp_adj = build_tp_adj(p.permute(1, 0, 2))
           
            tP_features = boxes_features.permute(1, 0, 2)
            for blk in self.Aggregation1:
                tP_features = blk(tP_features, tp_adj)
            # actions_feature = torch.clone(tP_features)
            sP_features = torch.mean(tP_features, dim=1).unsqueeze(0)
            sp = torch.mean(p, dim=1).unsqueeze(0)
            sp_adj = torch.mean(sp_adj, dim=0).unsqueeze(0)
            # sP_features = tP_features.permute(1, 0, 2)
            for blk in self.Aggregation2:
                sP_features = blk(sP_features, sp_adj)
            
            boxes_states = self.dpi_nl(sP_features)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)
            NFS = NFG
            # boxes_states = boxes_states.view(T, N, -1)

            # Predict actions
            actn_score = self.fc_actions(tP_features)  # T,N, actn_num
            actn_score = torch.mean(actn_score, dim=1).reshape(N, -1)  # N, actn_num
            actions_scores.append(actn_score)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim = 0)  # T, NFS
            hash_out, acty_score = self.head(boxes_states_pooled)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            hash_out = torch.mean(hash_out, dim=0).reshape(1, -1)
            activities_scores.append(acty_score)
            hash_code.append(hash_out)
            sp_ = self.gcn(actn_score, sp_adj.clone().mean(0))
            sp_ = torch.mean(sp_, dim=0).unsqueeze(0)
            sp_feat.append(sp_)
        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num
        hash_code = torch.cat(hash_code,dim=0)
        sp_feat = torch.cat(sp_feat, dim=0)
        return actions_scores,activities_scores, hash_code, sp_feat