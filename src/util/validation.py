import numpy as np
import torch


def get_start_position(pred: torch.Tensor) -> int:
    pred_np: np.ndarray = pred.detach().numpy()
    result = []
    for labels in pred_np:
        try:
            result.append(labels.tolist().index(1))
        except ValueError:
            result.append(0)
    return result


def evaluate(model, dev_dataloader):
    pred = []
    gold = []
    with torch.no_grad():
        for batch in dev_dataloader:
            if len(batch) == 2:
                input_ids, labels = batch
                labels = get_start_position(labels)
                predicted = model.predict(input_ids)
                pred.extend(predicted)
                gold.extend(labels)
            else:
                input_ids, _, words, _, true_labels = batch
                true_labels = true_labels.tolist()
                predicted = model.predict(input_ids, words)
                pred.extend(predicted)
                gold.extend(true_labels)

    distances = [abs(pred[i] - gold[i]) for i in range(len(pred))]
    mean = sum(distances) / len(distances)
    return mean
