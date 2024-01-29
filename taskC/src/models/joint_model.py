import torch
import torch.nn as nn
import torch.nn.functional as F

from models.char_bilstm import ChariBiLSTM, CharBiLSTMTokenizer
from models.word2vec_bilstm import Word2VecBiLSTM, Word2VecTokenizer

class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        

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
        self.char_model = ChariBiLSTM.from_pretrained(char_model)
        
        self.word2vec_tokenizer = Word2VecTokenizer.from_pretrained(
            word2vec_tokenizer, max_len=word2vec_max_len
        )
        self.word2vec_model = Word2VecBiLSTM.from_pretrained(word2vec_model)
        
        self.dim = self.char_model.hidden_size + self.word2vec_model.hidden_size
        

        
    def prepare(self, texts, labels):
        (
            char_ids, 
            char_labels, 
            char_words, 
            char_attentions
        ) = self.char_tokenizer.tokenize(texts, labels)
        
        (
            word2vec_ids, 
            word2vec_labels, 
            word2vec_words, 
            word2vec_attentions
        ) = self.word2vec_tokenizer.tokenize(texts, labels)
        
        char_out = self.char_model(char_ids)
        word2vec_out = self.word2vec_model(word2vec_ids)
        
        batch_size = char_out.shape[0]
        seq_len = len(torch.unique(char_words))
        
        char_vectors = torch.zeros(batch_size, seq_len, self.char_model.hidden_size)
        word2vec_vectors = torch.zeros(batch_size, seq_len, self.word2vec_model.hidden_size)
        labels = torch.zeros(batch_size, seq_len)
        
        for b in range(batch_size):
            for word_id in range(seq_len):
                cc_word_idx = torch.where(char_words[b] == word_id)[0]
                print(cc_word_idx)
                char_emb = torch.mean(char_out[b, cc_word_idx], dim=0)
                
                w2v_word_idx = torch.where(word2vec_words[b] == word_id)[0]
                print(w2v_word_idx)
                word2vec_emb = torch.mean(word2vec_out[b, w2v_word_idx], dim=0)
                
                char_vectors[b, word_id] = char_emb
                word2vec_vectors[b, word_id] = word2vec_emb
                boundary = labels[b]
                if word_id < boundary or boundary == -1:
                    labels[b, word_id] = 0
                else:
                    labels[b, word_id] = 1
        
        return char_vectors, word2vec_vectors, labels
    
    @staticmethod
    def collate_fn(tokenizer):
        def collate(batch):
            texts = [x[0] for x in batch]
            true_labels = [x[1] for x in batch]
            prepared = tokenizer.prepare(texts, true_labels)
            return prepared
        return collate
        
        
                
        
        