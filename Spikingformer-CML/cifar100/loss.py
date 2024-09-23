import torch
import torch.nn as nn
import torch.nn.functional as F

class BRDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """

    def __init__(self,
                 student_emb,
                 teacher_emb,
                 alpha_mgd=0.00007,
                 lambda_mgd=0.15,
                 use_clip=True,
                 ):
        super(BRDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        if student_emb != teacher_emb:
            self.align = nn.Conv2d(student_emb, teacher_emb, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.use_clip = use_clip

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_emb, teacher_emb, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_emb, teacher_emb, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*D*H*W, student's feature map
            preds_T(Tensor): Bs*D, teacher's feature map
        """
        # assert preds_S.shape[-1:] == preds_T.shape[-1:]
        if self.align is not None:
            preds_S = self.align(preds_S)

        if self.use_clip:
            # preds_T = torch.clip(preds_T, preds_T.min(), preds_S.max())
            preds_T = preds_T / (preds_T.max()) * preds_S.max()

        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, D, H, W = preds_S.shape

        device = preds_S.device
        mat = torch.rand((N, D, 1, 1)).to(device)
        # mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)
        new_fea = new_fea.flatten(2).mean(2)
        # print(new_fea.shape, preds_T.shape)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

class MSE(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """

    def __init__(self,
                 student_emb,
                 teacher_emb,
                 ):
        super(MSE, self).__init__()

        if student_emb != teacher_emb:
            self.align = nn.Conv2d(student_emb, teacher_emb, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*D*H*W, student's feature map
            preds_T(Tensor): Bs*D, teacher's feature map
        """
        # assert preds_S.shape[-1:] == preds_T.shape[-1:]
        if self.align is not None:
            preds_S = self.align(preds_S)
        preds_S = preds_S.flatten(2).mean(2)

        loss = nn.MSELoss()(preds_S, preds_T)

        return loss

def get_logits_loss(fc_t, fc_s, one_hot_label, temp, num_classes=1000):
    s_input_for_softmax = fc_s / temp
    t_input_for_softmax = fc_t / temp

    softmax = torch.nn.Softmax(dim=1)
    logsoftmax = torch.nn.LogSoftmax()

    t_soft_label = softmax(t_input_for_softmax)

    softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

    fc_s_auto = fc_s.detach()
    fc_t_auto = fc_t.detach()
    log_softmax_s = logsoftmax(fc_s_auto)
    log_softmax_t = logsoftmax(fc_t_auto)
    # one_hot_label = F.one_hot(label, num_classes=num_classes).float()
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss

    soft_loss = (temp ** 2) * torch.mean(softmax_loss)

    return soft_loss