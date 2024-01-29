from dataclasses import dataclass

from regex import D
from loader.data import TaskC_Data
from models.joint_model import JointModel, JointModelPreprocessor
from torch.utils.data import DataLoader


def add_args(parser):
    group = parser.add_argument_group("joint-model")
    def p(cmd): return f"--joint-model-{cmd}"
    args = [
        ("char-tokenizer", str, None),
        ("char-model", str, None),
        ("char-max-len", int, 10000),
        ("w2v-tokenizer", str, None),
        ("w2v-model", str, None),
        ("w2v-max-len", int, 10000),

        ("load-model", str, None),
        ("save-every", int, 1000),
        ("checkpoint-prefix", str, "joint_model"),
        ("epochs-ext", int, 10),
        ("epochs-pure", int, 10),
        ("lr", float, 1e-3),
        ("batch-size", int, 8),
        ("hidden-size", int, 2),
        ("dropout", float, 0.2),
    ]

    for arg in args:
        group.add_argument(p(arg[0]), type=arg[1], default=arg[2])

    group.add_argument(p("train"), action="store_true", default=False)


def evaluate(model, dev_loader):
    pass


@dataclass
class TrainingArguments:
    pass


def train(args):
    def arg(name):
        return getattr(args, "joint_model_" + name.replace("-", "_"))

    preprocessor = JointModelPreprocessor(
        char_model=arg("char-model"),
        char_tokenizer=arg("char-tokenizer"),
        char_max_len=arg("char-max-len"),
        word2vec_model=arg("w2v-model"),
        word2vec_tokenizer=arg("w2v-tokenizer"),
        word2vec_max_len=arg("w2v-max-len"),
    )

    train_ds_ext = TaskC_Data(split="train")
    train_ds_ext.import_task_A()
    train_ds_pure = TaskC_Data(split="train")
    dev_ds = TaskC_Data(split="dev")

    train_dl_ext = DataLoader(
        train_ds_ext,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
    )
    train_dl_pure = DataLoader(
        train_ds_pure,
        batch_size=arg("batch-size"),
        shuffle=True,
        collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=arg("batch-size"),
        shuffle=False,
        collate_fn=JointModelPreprocessor.collate_fn(preprocessor)
    )

    for batch in train_dl_pure:
        print(batch)
        break
