from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch


def evaluate(model, dev_dataloader, f1_only=True):
    y_pred = []
    y_gold = []
    with torch.no_grad():
        for input_ids, attentions, labels in dev_dataloader:
            _, out = model(input_ids, attentions)
            for i in range(out.shape[0]):
                pred = torch.argmax(out[i]).item()
                y_pred.append(pred)
                y_gold.append(labels[i].item())

    r = classification_report(y_gold, y_pred, zero_division=0.0)
    _, _, f1, _ = precision_recall_fscore_support(
        y_gold, y_pred, average="macro", zero_division=0.0)

    if f1_only:
        return f1

    return f1, r
