import json
import torch


class SpacyFeatures:
    def __init__(self, train_path, dev_path, train_ppl_path, dev_ppl_path, del_feats=None) -> None:
        if del_feats is not None:
            self.del_feats = [int(i) for i in del_feats.split(",")]
        else:
            self.del_feats = []

        self.order = self._get_order(train_path)

        self.train_ids, self.train_vectors = self.__load_data(
            train_path, train_ppl_path)
        self.dev_ids, self.dev_vectors = self.__load_data(
            dev_path, dev_ppl_path)
        self.dim = self.train_vectors.shape[1]

    def _get_order(self, train_file):
        with open(train_file, "r") as f:
            data = json.load(f)
            keys = list(data.keys())
            keys.sort()
            keys.append("ppl")
            del_keys = ["text", "label", "passed_quality_check",
                        "per_word_perplexity", "perplexity", "oov_ratio"]
            keys = [key for key in keys if key not in del_keys]
            return keys

    def __load_data(self, path, ppl_path):
        ids, vectors = {}, []
        feats = None
        ppl = None
        with open(path, "r") as f:
            feats = json.load(f)
        with open(ppl_path, "r") as f:
            ppl = json.load(f)

        for i, _id in enumerate(feats["id"]):
            vec = []
            for key in self.order:
                if key == "ppl":
                    vec.append(float(ppl["ppl"][_id]))
                else:
                    v = feats[key][_id] if feats[key][_id] is not None else 0
                    v = float(v)
                    vec.append(v)
            vectors.append(vec)
            ids[_id] = i
        return ids, torch.tensor(vectors, dtype=torch.float32)

    def scale(self):
        mean = self.train_vectors.mean(dim=0)
        std = self.train_vectors.std(dim=0)

        # indices where mean is inf:
        # inf_indices = torch.where(torch.isinf(mean))
        # print(inf_indices)

        self.train_vectors = (self.train_vectors - mean) / std
        self.dev_vectors = (self.dev_vectors - mean) / std

    def get(self, text_id, split="train"):
        text_id = str(text_id)
        if split == "train":
            try:
                return self.train_vectors[self.train_ids[text_id]]
            except:
                print("[Warining] falling back to zero vector, key = ", text_id)
                return torch.zeros(self.dim, dtype=torch.float32)
        else:
            try:
                return self.dev_vectors[self.dev_ids[text_id]]
            except:
                print("[Warining] falling back to zero vector, key = ", text_id)
                return torch.zeros(self.dim, dtype=torch.float32)
