from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from hyptorch.pmath import dist_matrix


class NCALoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.tau = args.tau
        self.hyp_c = args.hyp_c
        if self.hyp_c == 0:
            # Eq.4:
            # self.dist_f = lambda x, y: x @ y.t()  # code
            self.dist_f = lambda x, y: x @ y.t() * 2 - 2  # paper
        else:
            # Eq. 2
            self.dist_f = lambda x, y: -dist_matrix(x, y, c=self.hyp_c)

    def forward(self, batch, labels):
        """
        z = model(x).view(len(x) // cfg.num_samples, cfg.num_samples, cfg.emb)
        loss = 0
        for i in range(cfg.num_samples):
            for j in range(cfg.num_samples):
                if i != j:
                    l = loss_f(z[:, i], z[:, j])
                    loss += l

        # x0 and x1 - positive pair: x0=[?] x1=[?]
        bsize = x0.shape[0]
        eye_mask = torch.eye(bsize).cuda() * 1e9
        target = torch.arange(bsize).cuda()
        logits00 = self.dist_f(x0, x0) / self.tau - eye_mask
        logits01 = self.dist_f(x0, x1) / self.tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        """
        target = (labels @ labels.T > 0).float()
        logits = self.dist_f(batch, batch)  # Note: use similarity here not distance

        # loss = F.cross_entropy(logits / self.tau, target)  # <- buggy NCA

        logits.fill_diagonal_(torch.finfo(logits.dtype).min)  # no i in Ci: see Eq. 2 in paper ProxyNCA++
        exp = F.softmax(logits / self.tau, dim=1)
        exp = torch.sum(exp * target, dim=1)  # <- is NCA!
        non_zero = exp != 0
        loss = -torch.log(exp[non_zero]).mean()

        return loss


if __name__ == "__main__":
    from _utils import gen_test_data
    from pytorch_metric_learning.losses import NCALoss as PML_NCALoss

    embeddings, singles, onehots = gen_test_data(128, 10, 32, False)
    embeddings = F.normalize(embeddings)

    args = Namespace(tau=0.1, hyp_c=0)

    # loss0 = NCALoss(args)(embeddings, onehots)
    target = (onehots @ onehots.T > 0).float()
    logits = 2 * embeddings @ embeddings.T - 2

    # if we use CE, which is not NCA
    loss_ce = F.cross_entropy(logits / args.tau, target)

    # calc loss_ce step by step
    exp = F.softmax(logits / args.tau, dim=1)
    neg_log = -torch.log(exp)  # <- not NCA!
    loss_ce_step = (neg_log * target).sum(1).mean()

    # NCA from pytorch_metric_learning: only supports single label (categorical id)
    loss_pml = PML_NCALoss(1 / args.tau)(embeddings, singles)

    # calc loss_pml step by step
    _logits = logits.clone()
    _logits.fill_diagonal_(torch.finfo(_logits.dtype).min)
    exp = F.softmax(_logits / args.tau, dim=1)
    exp = torch.sum(exp * target, dim=1)  # <- is NCA!
    non_zero = exp != 0
    loss_pml_step = -torch.log(exp[non_zero]).mean()

    print(loss_ce, loss_ce_step, loss_pml, loss_pml_step)
