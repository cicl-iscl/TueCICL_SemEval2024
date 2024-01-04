import os

import torch
from .validation import evaluate


class ProgressTracker:
    def __init__(self, prefix) -> None:
        self.progress = {}
        self.prefix = prefix

        try:
            os.makedirs(f"checkpoints/{prefix}")
        except:
            pass

    def for_steps(self, model, dev_loader):
        if "best" not in self.progress:
            self.progress["best"] = 9999
        metric = evaluate(model, dev_loader)
        is_best = metric < self.progress["best"]
        cp = {
            "model": model.state_dict(),
            "metric": metric
        }
        if is_best:
            self.progress["best"] = metric
            torch.save(cp, f"checkpoints/{self.prefix}/best.pt")
        torch.save(cp, f"checkpoints/{self.prefix}/latest.pt")
        return self.progress["best"], metric

    def for_epoch(self, model, optimizer, epoch, dev_loader):
        metric = evaluate(model, dev_loader)
        cp = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric
        }
        fname = f"checkpoints/{self.prefix}/epoch_{epoch}.pt"
        torch.save(cp, fname)
