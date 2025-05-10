from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import lightning as L
import module.ptb as ptb
import re
import collections
import pickle as pkl
from tqdm import tqdm
import numpy as np
import os

# torchtext.disable_torchtext_deprecation_warning()

class HuggingDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data",
            batch_size: int = 32,
            context_size: int = 3,
            num_workers: int = 8,
            device: str = 'cpu'
            ):
        super().__init__()
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.context_size: int = context_size
        self.num_workers: int = num_workers
        self.my_device: str = device
        self.datasets_name: str = "haukur/enwik9"
        self.datasets = None
        self.vocab_size: int = 0
        self.power: float = 0.75
        self.word_probs: torch.Tensor = None
        self.context: torch.Tensor = None
        self.target: torch.Tensor = None
        self.prepare_data_flag: bool = False
        self.setup_flag: bool = False
        self.word_to_id: dict = {}
        self.id_to_word: dict = {}
        self.corpus: list = []

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return {
            'context': self.context[idx],
            'target': self.target[idx]
        }

    def prepare_data(self):
        if self.prepare_data_flag:
            return
        os.makedirs(self.data_dir, exist_ok=True)
        self.datasets = load_dataset(self.datasets_name, cache_dir=self.data_dir, split='train')
        print("Finished downloading the dataset")
        self.prepare_data_flag = True

    def setup(self, stage: str=None):
        if self.setup_flag:
            return
        print("Setting up the dataset")
        os.makedirs(os.path.join(self.data_dir, f'datasets/setup_data/{self.datasets_name}'), exist_ok=True)
        pkl_wordid = os.path.join(self.data_dir, f'datasets/setup_data/{self.datasets_name}/wordid.pkl')
        pkl_setup = os.path.join(self.data_dir, f'datasets/setup_data/{self.datasets_name}/setup_data_{self.context_size}.pkl')
        print("Checking for the existence of [wordid.pkl]")

        # Check if wordid.pkl exists
        # If it exists, load the wordid.pkl file
        if os.path.isfile(pkl_wordid):
            print("Successfully confirmed the existence of [wordid.pkl]")
            print(f"Starting to load [wordid.pkl]")
            with open(pkl_wordid, "rb") as f:
                save_data = pkl.load(f)
            self.word_to_id = save_data['word_to_id']
            self.id_to_word = save_data['id_to_word']
            self.corpus = save_data['corpus_id']
            tokenized_data = save_data['corpus_word']
            print(f"Finished loading [wordid.pkl]")

        # If it does not exist, process the data and save it to wordid.pkl
        else:
            print("[wordid.pkl] was not found")
            print("Processing [wordid.pkl]")
            tokenized_data = self.data_organization()
            self.one_hot(tokenized_data)
            save_data = {
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'corpus_id': self.corpus,
                'corpus_word': tokenized_data
            }
            self.wordid_load = True
            with open(pkl_wordid, 'wb') as f:
                pkl.dump(save_data, f)
            print("Completed processing [wordid.pkl]")

        print(f"Checking for the existence of [setup_data_{self.context_size}.pkl]")
        self.vocab_size = len(self.word_to_id)

        # Check if setup_data_{context_size}.pkl exists
        # If it exists, load the setup_data_{context_size}.pkl file
        if os.path.isfile(pkl_setup):
            print(f"Successfully confirmed the existence of [setup_data_{self.context_size}.pkl]")
            print(f"Starting to load [setup_data_{self.context_size}.pkl]")
            with open(pkl_setup, "rb") as f:
                save_data = pkl.load(f)
                self.context = save_data['context']
                self.target = save_data['target']
                self.word_probs = save_data['word_probs']
            print(f"Finished loading [setup_data_{self.context_size}.pkl]")
        
        # If it does not exist, process the data and save it to setup_data_{context_size}.pkl
        else:
            print(f"[setup_data_{self.context_size}.pkl] was not found")
            print(f"Processing [setup_data_{self.context_size}.pkl]")

            self.create_context_target()

            with open(pkl_setup, "wb") as f:
                save_data = {
                    'context': self.context,
                    'target': self.target,
                    'word_probs': self.word_probs
                }
                pkl.dump(save_data, f)
            print(f"Completed processing [setup_data_{self.context_size}.pkl]")
        
        self.context.to('cpu')
        self.target.to('cpu')
        self.word_probs.to('cpu')

        print("Finished setting up the dataset")
        self.setup_flag = True

    # Data organization and cleaning
    def data_organization(self):
        text_raw = 10000

        # data = [re.sub(r"<.*?>", "", item['text'])
        #         for i,item in enumerate(tqdm(self.datasets, desc="Loading data"))
        #         if i < text_raw
        #         ]

        data = [re.sub(r"<.*?>", "", item['text'])
                for i,item in enumerate(tqdm(self.datasets, desc="Loading data"))
                ]
        
        tokenized_data = []
        for text in tqdm(data, desc="Cleaning data"):
            text = re.sub(r"[^a-zA-Z0-9\s.,?]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = text.lower()
            text = text.replace('.', ' .')
            text = text.replace(',', ' ,')
            text = text.replace('?', ' ?')
            text = text.split()
            tokenized_data.append(text)
        tokenized_data = [item for sublist in tqdm(tokenized_data, desc="Organizing data") for item in sublist]
        return tokenized_data

    # One-hot encoding
    def one_hot(self, tokenized_data):
        for word in tqdm(tokenized_data, desc="One-hot encoding"):
            if word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word
            self.corpus.append(self.word_to_id[word])
    
    # Create context and target
    def create_context_target(self):
        num_word = torch.zeros(self.vocab_size)
        self.context = torch.zeros(len(self.corpus) - 2*self.context_size, self.context_size*2, dtype=torch.long)
        self.target = torch.zeros(len(self.corpus) - 2*self.context_size, dtype=torch.long)
        for i in tqdm(range(len(self.corpus)), desc="Preparing context and target"):
            num_word[self.corpus[i]] += 1
            if i < self.context_size or i >= len(self.corpus) - self.context_size: continue
            index = i - self.context_size
            for j in range(self.context_size):
                self.context[index, j] = self.corpus[index + j]
            for j in range(self.context_size):
                self.context[index, self.context_size + j] = self.corpus[i + 1 + j]
            self.target[index] = self.corpus[i]
        num_word = torch.pow(num_word, self.power)
        total_word = num_word.sum()
        self.word_probs = num_word / total_word

    # DataLoader
    def train_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
            )
    

class PtbDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data",
            batch_size: int = 32,
            context_size: int =3,
            num_workers: int = 8,
            device: str = 'cpu'
            ):
        super().__init__()
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.context_size: int = context_size
        self.num_workers: int = num_workers
        self.my_device: str = device
        self.vocab_size: int = 0
        self.power: float = 0.75
        self.word_probs: torch.Tensor = None
        self.context: torch.Tensor = None
        self.target: torch.Tensor = None
        self.prepare_data_flag: bool = False
        self.setup_flag: bool = False
        self.word_to_id: dict = {}
        self.id_to_word: dict = {}
        self.corpus: list = []

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return {
            'context': self.context[idx],
            'target': self.target[idx]
        }

    def prepare_data(self):
        if self.prepare_data_flag:
            return
        os.makedirs(self.data_dir, exist_ok=True)
        self.corpus, self.word_to_id, self.id_to_word = ptb.load_data('train')
        print("Finished downloading the dataset")
        self.prepare_data_flag = True

    def setup(self, stage: str=None):
        if self.setup_flag:
            return
        print("Setting up the dataset")
        self.vocab_size = len(self.word_to_id)
        self.context, self.target = self.create_contexts_target(self.corpus, self.context_size)
        self.create_probs()

        self.word_probs = torch.tensor(self.word_probs, dtype=torch.float32)
        self.context = torch.tensor(self.context, dtype=torch.long)
        self.target = torch.tensor(self.target, dtype=torch.long)
        self.context.to('cpu')
        self.target.to('cpu')
        self.word_probs.to('cpu')

        print("Finished setting up the dataset")
        self.setup_flag = True
    
    # Create context and target
    def create_contexts_target(self, corpus, window_size):
        '''コンテキストとターゲットの作成

        :param corpus: コーパス（単語IDのリスト）
        :param window_size: ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
        :return:
        '''
        target = corpus[window_size:-window_size]
        contexts = []

        for idx in range(window_size, len(corpus)-window_size):
            cs = []
            for t in range(-window_size, window_size + 1):
                if t == 0:
                    continue
                cs.append(corpus[idx + t])
            contexts.append(cs)

        return np.array(contexts), np.array(target)
    
    def create_probs(self):
        counts = collections.Counter()
        for word_id in self.corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_probs = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_probs[i] = counts[i]

        self.word_probs = np.power(self.word_probs, self.power)
        self.word_probs /= np.sum(self.word_probs)

    # DataLoader
    def train_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
            )