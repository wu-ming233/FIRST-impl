import torch
import torch.nn as nn

class WeightedRankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, rankings):
        loss = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if rankings[i] < rankings[j]:
                    weight = 1 / (rankings[i] + rankings[j])
                    pair_loss = torch.log(1 + torch.exp(logits[j] - logits[i]))
                    loss += weight * pair_loss
        return loss / batch_size

class FIRSTLoss(nn.Module):
    def __init__(self, lm_loss_weight=1.0, rank_loss_weight=10.0, ignore_index=-100):
        super().__init__()
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.rank_loss = WeightedRankNetLoss()
        self.lm_loss_weight = lm_loss_weight
        self.rank_loss_weight = rank_loss_weight
        self.ignore_index = ignore_index

    def forward(self, lm_logits, lm_labels, ranking_logits, rankings):
        lm_loss = self.lm_loss(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
        rank_loss = self.rank_loss(ranking_logits, rankings)
        return self.lm_loss_weight * lm_loss + self.rank_loss_weight * rank_loss