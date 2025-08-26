import torch

class Accuracy:
    def __init__(self, top_k=1):
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, targets):
        with torch.no_grad():
            if self.top_k == 1:
                preds = logits.argmax(dim=1)
                self.correct += (preds == targets).sum().item()
                self.total += targets.numel()
            else:
                topk = logits.topk(self.top_k, dim=1).indices
                self.correct += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
                self.total += targets.size(0)

    def compute(self):
        if self.total == 0: return 0.0
        return float(self.correct) / float(self.total)
