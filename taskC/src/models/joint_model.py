import torch
import torch.nn as nn
import torch.nn.functional as F

from models.char_bilstm import ChariBiLSTM, CharBiLSTMTokenizer
from models.word2vec_bilstm import Word2VecBiLSTM, Word2VecTokenizer
from util.device import get_device
from util.time import timestart

class JointModel(nn.Module):
    def __init__(self, char_size=None, w2v_size=None, hidden_size=None, dropout=None) -> None:
        super().__init__()
        
        self.char_size = char_size
        self.w2v_size = w2v_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.char_w = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.w2v_w = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.input_size = self.char_size + self.w2v_size
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 2),
        )
        
    def to_device(self):
        self.char_w.to(get_device())
        self.w2v_w.to(get_device())
        self.mlp.to(get_device())
    
    def forward(self, X_char, X_w2v):
        # X_char: (batch_size, seq_len, char_size)
        # X_w2v: (batch_size, seq_len, w2v_size)
        
        # Apply weights
        X_char = self.char_w * X_char
        X_w2v = self.w2v_w * X_w2v
        
        # Concatenate to get shape: (batch_size, seq_len, char_size + w2v_size)
        X = torch.cat((X_char, X_w2v), dim=2)
        # Apply MLP
        out = self.mlp(X)
        
        return F.log_softmax(out, dim=-1)
    
    def save(self, path, extra):
        save_data = {
            "state_dict": self.state_dict(),
            "char_size": self.char_size,
            "w2v_size": self.w2v_size,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            **extra
        }
        torch.save(save_data, path)
    
    @classmethod
    def from_pretrained(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        model = JointModel(
            char_size=checkpoint["char_size"],
            w2v_size=checkpoint["w2v_size"],
            hidden_size=checkpoint["hidden_size"],
            dropout=checkpoint["dropout"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to_device()
        return model, checkpoint
        

class JointModelPreprocessor:
    def __init__(
        self,
        char_model: ChariBiLSTM = None,
        char_tokenizer: CharBiLSTMTokenizer = None,
        word2vec_model: Word2VecBiLSTM = None,
        word2vec_tokenizer: Word2VecTokenizer = None,
        char_max_len: int = 10000,
        word2vec_max_len: int = 10000,
    ) -> None:
        
        self.char_tokenizer = CharBiLSTMTokenizer.from_pretrained(
            char_tokenizer, max_len=char_max_len
        )
        self.char_model, _ = ChariBiLSTM.from_pretrained(char_model)
        self.char_model.to_device()
        self.char_model.eval()
        
        self.word2vec_tokenizer = Word2VecTokenizer.from_pretrained(
            word2vec_tokenizer, max_len=word2vec_max_len
        )
        self.word2vec_model, _ = Word2VecBiLSTM.from_pretrained(word2vec_model)
        self.word2vec_model.to_device()
        self.word2vec_model.eval()
        
        self.dim = self.char_model.hidden_size + self.word2vec_model.hidden_size
        
        
    def prepare(self, texts, true_labels):
        (
            char_ids, 
            char_labels, 
            char_words, 
            char_attentions
        ) = self.char_tokenizer.tokenize(texts, true_labels)
        
        (
            word2vec_ids, 
            word2vec_labels, 
            word2vec_words, 
            word2vec_attentions
        ) = self.word2vec_tokenizer.tokenize(texts, true_labels)
        
        with torch.no_grad():
            
            char_ids = char_ids.to(get_device())
            _, char_out = self.char_model(char_ids, return_hidden=True)
            # do not move w2v ids to device, since Embedding is on cpu
            _, word2vec_out = self.word2vec_model(word2vec_ids, return_hidden=True)
                        
            batch_size = char_out.shape[0]
            seq_len = len(torch.unique(char_words))
            
            char_vectors = torch.zeros(
                (batch_size, seq_len, self.char_model.hidden_size * 2), 
                dtype=torch.float32, 
                requires_grad=False, 
                device=get_device()
            )
            word2vec_vectors = torch.zeros(
                (batch_size, seq_len, self.word2vec_model.hidden_size * 2), 
                dtype=torch.float32, 
                requires_grad=False, 
                device=get_device()
            )
            labels = torch.zeros(
                (batch_size, seq_len), 
                dtype=torch.long, 
                requires_grad=False, 
                device=get_device()
            )
            
            attentions = torch.full(
                (batch_size, seq_len),
                1,
                dtype=torch.long,
                requires_grad=False,
                device=get_device()
            )
            
            
            for b in range(batch_size):
                cc_attn = char_attentions[b].cpu().tolist()
                cc_attn_bound = cc_attn.index(0) if 0 in cc_attn else len(cc_attn)
                cc_words = char_words[b][:cc_attn_bound]
                
                w2v_attn = word2vec_attentions[b].cpu().tolist()
                w2v_attn_bound = w2v_attn.index(0) if 0 in w2v_attn else len(w2v_attn)
                w2v_words = word2vec_words[b][:w2v_attn_bound]
                
                for word_id in range(seq_len):
                    cc_word_idx = torch.where(cc_words == word_id)[0]
                    if len(cc_word_idx) == 0:
                        char_emb = torch.zeros(
                            self.char_model.hidden_size * 2, 
                            dtype=torch.float32, 
                            requires_grad=False,
                            device=get_device()
                        )
                    else:
                        char_emb = torch.mean(char_out[b, cc_word_idx], dim=0)
                    
                    w2v_word_idx = torch.where(w2v_words == word_id)[0]
                    if len(w2v_word_idx) == 0:
                        # we have reached padding
                        word2vec_emb = torch.zeros(
                            self.word2vec_model.hidden_size * 2, 
                            dtype=torch.float32, 
                            requires_grad=False,
                            device=get_device()
                        )
                        attentions[b, word_id] = 0
                    else:
                        word2vec_emb = torch.mean(word2vec_out[b, w2v_word_idx], dim=0)
                        attentions[b, word_id] = 1
                         
                    char_vectors[b, word_id] = char_emb
                    word2vec_vectors[b, word_id] = word2vec_emb
                    boundary = true_labels[b]
                    if word_id < boundary or boundary == -1:
                        labels[b, word_id] = 0
                    else:
                        labels[b, word_id] = 1
                                    
            return char_vectors, word2vec_vectors, labels, attentions
    
    @staticmethod
    def collate_fn(tokenizer):
        def collate(batch):
            texts = [x[0] for x in batch]
            true_labels = [x[1] for x in batch]
            prepared = tokenizer.prepare(texts, true_labels)
            return prepared
        return collate
        
        
                
        
        