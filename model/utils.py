import torch
import time
import numpy as np
import torchvision.transforms as transforms
from thop import profile, clever_format
import torch.nn as nn
import torch
import numpy as np
import pickle
'''
Thanks to https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py.
'''

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)

    return images


def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx=X.pow(2).sum(dim=1).reshape((-1,1))
    ry=Y.pow(2).sum(dim=1).reshape((-1,1))
    dist=rx-2.0*X.matmul(Y.t())+ry.t()
    return torch.sqrt(dist)

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B=X.shape[0]
    
    rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
    ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
    dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
    
    return torch.sqrt(dist)

def sincos_encoding_2d(positions,d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N=positions.shape[0]
    
    d=d_emb//2
    
    idxs = [np.power(1000,2*(idx//2)/d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)
    
    idxs = idxs.repeat(N,2)  #N, d_emb
    
    pos = torch.cat([ positions[:,0].reshape(-1,1).repeat(1,d),positions[:,1].reshape(-1,1).repeat(1,d) ],dim=1)

    embeddings=pos/idxs
    
    embeddings[:,0::2]=torch.sin(embeddings[:,0::2])  # dim 2i
    embeddings[:,1::2]=torch.cos(embeddings[:,1::2])  # dim 2i+1
    
    return embeddings


def print_log(file_path,*args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f)

def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k,v in cfg.__dict__.items():
        print_log(cfg.log_path, k,': ',v)
    print_log(cfg.log_path, '======================End=======================')
    
def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase=='Test':
        print_log(log_path, '====> %s at epoch #%d'%(phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d'%(phase, info['epoch']))
        
    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds'%(
                info['activities_acc'], info['loss'], info['time']))

    if 'activities_conf' in info.keys():
        print_log(log_path, info['activities_conf'])
    if 'activities_MPCA' in info.keys():
        print_log(log_path, 'Activities MPCA:{:.2f}%'.format(info['activities_MPCA']))
    if 'MAD' in info.keys():
        print_log(log_path, 'MAD:{:.4f}'.format(info['MAD']))
    print_log(log_path, '\n')
        
def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg=['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                   'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                   'result_path', 'log_path', 'device']
    
    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)
        
        for k,v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print( k,': ',v, file=f)
            
        print('=====================Result======================', file=f)
        
        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)
            
        print('Cost total %.4f hours.'%(exp_result['total_time']), file=f)
        
        print('======================End=======================', file=f)
    
    
    data_dict=pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name]=exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))
        
    
class AverageMeter(object):
    """
    Computes the average value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time=time.time()
        
    def timeit(self):
        old_time=self.last_time
        self.last_time=time.time()
        return self.last_time-old_time

class ConfusionMeter(object):
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)

        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

def MPCA(conf_mat):
    '''

    :param conf_mat: np.ndarray((k, k), dtype=np.int32)
    :return:
    '''
    class_sum = np.sum(conf_mat, axis = 1, dtype = np.float32)
    for i in range(len(class_sum)):
        class_sum[i] = np.float32(conf_mat[i][i])/np.float32(class_sum[i])
    mpca = np.mean(class_sum)*100
    return mpca

def MAC2FLOP(macs, params, module_name = ''):
    macs, params = clever_format([macs, params], "%.3f")
    print('{} MACs: {}  #Params: {}'.format(module_name, macs, params))
    if 'M' in macs:
        flops = float(macs.replace('M', '')) * 2
        flops = str(flops/1000) + 'G'
    elif 'G' in macs:
        flops = str(float(macs.replace('G', '')) * 2) + 'G'
    print('{} GFLOPs: {}  #Params: {}'.format(module_name, flops, params))


