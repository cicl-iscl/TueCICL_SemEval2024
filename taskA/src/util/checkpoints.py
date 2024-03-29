import os

import torch

from util.core import abspath


class ProgressTracker:
    def __init__(self, prefix, evaluate_fn, last_epoch_only=False, save_latest=True) -> None:
        self.progress = {}
        self.prefix = prefix
        self.basedir = abspath(__file__, "../../checkpoints")
        self.evaluate_fn = evaluate_fn
        
        self.last_epoch_only = last_epoch_only
        self.save_latest = save_latest

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

    def save(self, model, path, extra):
        if hasattr(model, "module"):
            model.module.save(path, extra)
        else:
            model.save(path, extra)

    def for_steps(self, model, dev_loader):
        best_path = f"{self.basedir}/{self.prefix}/best.pt"
        metric = self.evaluate_fn(model, dev_loader)
        is_best = metric > self.progress["best"]
        extra = {
            "metric": metric
        }
        if is_best:
            self.progress["best"] = metric
            self.save(model, best_path, extra)
        if self.save_latest:
            self.save(model, f"{self.basedir}/{self.prefix}/latest.pt", extra)
        return self.progress["best"], metric

    def for_epoch(self, model, optimizer, epoch, dev_loader):
        metric = self.evaluate_fn(model, dev_loader)
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric
        }
        if self.last_epoch_only:
            fname = f"{self.basedir}/{self.prefix}/last_epoch.pt"
        else:
            fname = f"{self.basedir}/{self.prefix}/epoch_{epoch}.pt"
        self.save(model, fname, extra)
