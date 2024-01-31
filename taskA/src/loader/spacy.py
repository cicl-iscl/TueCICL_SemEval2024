import json
import torch


class SpacyFeatures:
    def __init__(self, train_path, dev_path, ppl_path_train, ppl_path_dev, del_feats=None) -> None:
        if del_feats is not None:
            self.del_feats = [int(i) for i in del_feats.split(",")]
        else:
            self.del_feats = []

        self.train_ids, self.train_vectors = self.__load_data(
            train_path, ppl_path_train)
        self.dev_ids, self.dev_vectors = self.__load_data(
            dev_path, ppl_path_dev)
        self.dim = self.train_vectors.shape[1]

    def __load_data(self, path, ppl_path):
        _ids, vectors = {}, []
        with open(ppl_path, 'r') as f:
            ppl = json.load(f)
        with open(path, 'r') as f:
            i = 0
            for line in f:
                try:
                    content = json.loads(line)
                    vec = content["vector"]
                    vec = [0.0 if not x else x for x in vec]
                    vec = [vec[j]
                           for j in range(len(vec)) if j not in self.del_feats]
                    vec.append(ppl["ppl"][str(content["id"])])
                    _id = content["id"]
                    _ids[_id] = i
                    vectors.append(
                        torch.tensor(vec, dtype=torch.float32)
                    )
                    i += 1
                except Exception as e:
                    raise e
        return _ids, torch.stack(vectors)

    def scale(self):
        mean = self.train_vectors.mean(dim=0)
        std = self.train_vectors.std(dim=0)

        # indices where mean is inf:
        # inf_indices = torch.where(torch.isinf(mean))
        # print(inf_indices)

        self.train_vectors = (self.train_vectors - mean) / std
        self.dev_vectors = (self.dev_vectors - mean) / std

    def get(self, text_id, split="train"):
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
