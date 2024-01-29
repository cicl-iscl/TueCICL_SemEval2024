import os

import torch


class ProgressTracker:
    def __init__(self, prefix, evaluate_fn, save_latest=False, last_epoch_only=True) -> None:
        self.progress = {}
        self.prefix = prefix
        self.evaluate_fn = evaluate_fn
        self.basedir = f"../checkpoints/{prefix}"

        try:
            os.makedirs(self.basedir)
        except:
            pass
    
    def _save(self, model, path, extra):
        try:
            _model = model.module if hasattr(model, "module") else model
            _model.save(path, extra)
        except:
            print("Failed to save model")

    def for_steps(self, model, dev_loader):
        best_path = f"{self.basedir}/best.pt"
        if "best" not in self.progress:
            try:
                last_best = torch.load(best_path)
                self.progress["best"] = last_best["metric"]
            except:
                self.progress["best"] = 9999
        metric = self.evaluate_fn(model, dev_loader)
        is_best = metric < self.progress["best"]
        extra = {
            "metric": metric
        }
        if is_best:
            self.progress["best"] = metric
            self._save(model, best_path, extra)
        if self.save_latest:
            self._save(model, f"{self.basedir}/latest.pt", extra)
        return self.progress["best"], metric

    def for_epoch(self, model, optimizer, epoch, dev_loader):
        metric = self.evaluate_fn(model, dev_loader)
        extra = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric
        }
        if self.save_latest:
            fname = f"{self.basedir}/epoch_{epoch}.pt"
        else:
            fname = f"{self.basedir}/last_epoch.pt"
        self._save(model, fname, extra)
