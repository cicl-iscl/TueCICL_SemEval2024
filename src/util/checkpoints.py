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
        best_path = f"checkpoints/{self.prefix}/best.pt"
        if "best" not in self.progress:
            try:
                last_best = torch.load(best_path)
                self.progress["best"] = last_best["metric"]
            except:
                print("No previous best found")
                self.progress["best"] = 9999
        metric = evaluate(model, dev_loader)
        is_best = metric < self.progress["best"]
        extra = {
            "metric": metric
        }
        if is_best:
            self.progress["best"] = metric
            model.save(best_path, extra)
        model.save(f"checkpoints/{self.prefix}/latest.pt", extra)
        return self.progress["best"], metric

    def for_epoch(self, model, optimizer, epoch, dev_loader):
        metric = evaluate(model, dev_loader)
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric
        }
        fname = f"checkpoints/{self.prefix}/epoch_{epoch}.pt"
        model.save(fname, extra)
