import os
from tqdm import tqdm
import torch
from module.datamodule_huggingface import HuggingDataModule, PtbDataModule
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from module.train_model import Train_model
from datetime import datetime
import pickle as pkl
import numpy as np
from time import time

def main():
    start = time()
    torch.set_float32_matmul_precision('medium')  # or 'high'
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data_dir = "data"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 8
    embedding_dim = 100
    batch_size = 100
    context_size = 5
    num_negative = 5
    learning_rate = 0.001
    epochs = 1

    # datamodule = HuggingDataModule(data_dir=data_dir, batch_size=batch_size, context_size=context_size, num_workers=num_workers, device=device)
    datamodule = PtbDataModule(data_dir=data_dir, batch_size=batch_size, context_size=context_size, num_workers=num_workers, device=device)
    datamodule.prepare_data()
    datamodule.setup()
    
    model = Train_model(
        datamodule.vocab_size,
        datamodule.word_probs,
        context_size=context_size,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        num_negative=num_negative,
        learning_rate=learning_rate,
        device=device
        )
    logger = TensorBoardLogger(save_dir="logs", name="CBOW_model", version=run_time)
    trainer = L.Trainer(max_epochs=epochs, accelerator=device, devices=1, logger=logger)
    trainer.fit(model, datamodule)

    wordvecs = model.model.embedding_w1.weight.detach().cpu().numpy()
    vecs_path = os.path.join(logger.log_dir,"wordvecs.pkl" )
    save_data ={
        'wordvecs' : wordvecs,
        'word_to_id': datamodule.word_to_id,
        'id_to_word': datamodule.id_to_word
    }
    with open(vecs_path, "wb") as f:
        pkl.dump(save_data, f)
    
    end = time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"processing time:{hours}h{minutes}m{seconds}s")


if __name__ == "__main__":
    main()