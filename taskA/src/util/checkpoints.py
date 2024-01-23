import os

import torch

from util.core import abspath


class ProgressTracker:
    def __init__(self, prefix, evaluate_fn) -> None:
        self.progress = {}
        self.prefix = prefix
        self.basedir = abspath(__file__, "../../checkpoints")
        self.evaluate_fn = evaluate_fn

        try:
            os.makedirs(f"{self.basedir}/{prefix}")
        except:
            pass
        
        if "best" not in self.progress:
            try:
                best_path = f"{self.basedir}/{self.prefix}/best.pt"
                last_best = torch.load(best_path)
                self.progress["best"] = last_best["metric"]
            except:
                self.progress["best"] = 0

    def for_steps(self, model, dev_loader):
        best_path = f"{self.basedir}/{self.prefix}/best.pt"
        metric = self.evaluate_fn(model, dev_loader)
        is_best = metric > self.progress["best"]
        extra = {
            "metric": metric
        }
        if is_best:
            self.progress["best"] = metric
            model.module.save(best_path, extra)
        model.module.save(f"{self.basedir}/{self.prefix}/latest.pt", extra)
        return self.progress["best"], metric

    def for_epoch(self, model, optimizer, epoch, dev_loader):
        metric = self.evaluate_fn(model, dev_loader)
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric
        }
        fname = f"{self.basedir}/{self.prefix}/epoch_{epoch}.pt"
        model.module.save(fname, extra)