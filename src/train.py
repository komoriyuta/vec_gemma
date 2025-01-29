# train.py
import os
import argparse
import logging
import random
from pathlib import Path
from huggingface_hub import snapshot_download
import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from datasets import disable_progress_bars
disable_progress_bars()
from datasets import load_dataset

from schedulefree import RAdamScheduleFree
from sentence_transformers import SentenceTransformer

from gemma.config import GemmaConfig, get_model_config
from gemma.model import GemmaForCausalLM
from VAEs.LinearVAE import LinearVAE
from VAEs.VAE import VAE

torch.set_float32_matmul_precision('high')




class TrainingConfig:
    def __init__(self):
        # データ設定
        self.batch_size = 4
        self.max_seq_len = 256
        self.num_workers = 4
        
        # 最適化設定
        self.lr = 1e-4
        self.num_epochs = 10
        self.grad_accum_steps = 4
        self.beta_init = 0.05
        self.beta_max = 0.4
        self.beta_step = 1e-6
        self.crop_lambda = 0.2

        # モデル設定
        self.bert_model_name = "cl-nagoya/ruri-large"
        self.gemma_model_size = "2b-v2"
        self.vae_hidden_dim = 512
        self.vae_latent_dim = 128
        
        # 生成設定
        self.sample_interval = 1000
        self.num_samples = 3
        self.max_gen_length = 100
        self.generation_temp = 0.1
        self.generation_top_p = 0.2
        self.generation_top_k = 2
        
        self.ckpt_interval = 5000
        # パス設定
        self.log_dir = f"./logs/{datetime.datetime.now()}/"
        self.checkpoint_dir = "./checkpoints"
        self.dataset_path = "AhmedSSabir/Japanese-wiki-dump-sentence-dataset"
        
        # 初期化
        self._setup_directories()
        
    def _setup_directories(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

def prepare_dataset(config):
    def collate_fn(batch):
        return [item['text'] for item in batch]
    
    dataset = load_dataset(config.dataset_path, cache_dir="./.datasets")

    train_loader = DataLoader(
        dataset['train'].with_format("torch"),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return train_loader

def initialize_models(config, device):
    # BERTモデル
    bert_model = SentenceTransformer(config.bert_model_name)
    bert_model.requires_grad_(False)
    bert_model = bert_model.to(device)
    
    # Gemmaモデル
    snapshot_dir = snapshot_download(repo_id='google/gemma-2-2b-jpn-it-pytorch')

    # Ensure that the tokenizer is present
    tokenizer_path = os.path.join(snapshot_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

    # Ensure that the checkpoint is present
    ckpt_path = os.path.join(snapshot_dir, f'model.ckpt')
    assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

    gemma_model_config = get_model_config("2b-v2")
    gemma_model_config.tokenizer = tokenizer_path
    
    # Instantiate the model and load the weights.
    torch.set_default_dtype(gemma_model_config.get_dtype())
    gemma_model = GemmaForCausalLM(gemma_model_config)
    gemma_model.requires_grad_(False)
    gemma_model.load_weights(ckpt_path)
    gemma_model = gemma_model.to(device).eval()
    # VAEモデル
    vae_model = LinearVAE(
        bert_model.get_sentence_embedding_dimension(),
        gemma_model.config.hidden_size,
        hidden_dim=config.vae_hidden_dim,
        latent_dim=config.vae_latent_dim
    ).to(device)
    
    return bert_model, gemma_model, vae_model

def generate_and_log_samples(vae_model, bert_model, gemma_model, device, writer, global_step, config):
    vae_model.eval()
    sample_texts = [
        "人工知能の未来について",
        "量子コンピュータの可能性",
        "ディープラーニングの応用分野",
        "自然言語処理の最新動向",
        "ロボット工学の進化",
        "一時期所長となる。",
        "京都府京都市に生まれる。",
        "卒業後は文章を書く仕事がしたいと1994年に報知新聞社にスポーツ記者として勤務し、高校野球やゴルフを取材する。",
        "その後、全てをリセットするためにタンザニア・ダルエスサラーム大学に留学し、スワヒリ語科で学ぶ。",
        "29歳の時に新人賞の最終候補に残るが、その後結婚し、出産したことで小説を書く余裕を無くしてしまう。",
        "本形式は、192形を改称して生まれた形式である。",
        "ただし、現実にはこの変更を受けたのは数両程度である。",  
    ]
    
    with torch.no_grad():
        prep_texts = ["文章: " + text for text in sample_texts]
        
        # BERTで埋め込みを取得
        bert_embeddings = bert_model.encode(
            prep_texts, 
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        ).to(torch.bfloat16)
        
        # VAEで埋め込みを変換
        vae_output, _, _ = vae_model(bert_embeddings)
        
        # Gemmaでテキスト生成
        generated_texts = []
        for i in range(len(sample_texts)):
            try:
                embedding = vae_output[i].unsqueeze(0).unsqueeze(0)
                generated = gemma_model.generate_with_initial_embedding(
                    initial_embedding=embedding,
                    device=device,
                    output_len=config.max_gen_length,
                    temperature=config.generation_temp,
                    top_p=config.generation_top_p,
                    top_k=config.generation_top_k
                )
                generated_texts.append(generated[0])
            except Exception as e:
                generated_texts.append(f"生成エラー: {str(e)}")
        
        # ログに記録
        log_text = "\n\n".join([
            f"入力: {sample_texts[i]}\n生成結果: {generated_texts[i]}" 
            for i in range(len(sample_texts))
        ])
        
        writer.add_text("生成サンプル", log_text, global_step)
        logging.info(f"\n=== ステップ {global_step} の生成サンプル ===\n{log_text}\n")
    
    vae_model.train()

def train(config, vae_model, bert_model, gemma_model, optimizer, device, start_epoch=0, initial_beta=0.1):
    writer = SummaryWriter(config.log_dir)
    setup_logging(config.log_dir)
    
    #scaler = torch.amp.GradScaler()
    best_loss = float('inf')
    beta = initial_beta
    optimizer.train()

    for epoch in range(start_epoch, config.num_epochs):
        logging.info(f"エポック {epoch+1}/{config.num_epochs}")
        train_loader = prepare_dataset(config)
          # DataLoaderをエポックごとに再作成
        total_steps = len(train_loader)
        progress_bar = tqdm(
            train_loader,
            total=total_steps,
            desc=f"エポック {epoch+1}/{config.num_epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            dynamic_ncols=True
        )
        for step, batch_texts in enumerate(progress_bar):
            with torch.no_grad():
                rep_texts = ["文章: " + text for text in batch_texts]
                bert_embeddings = bert_model.encode(
                    rep_texts, 
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False
                ).to(torch.bfloat16)
            
            # 混合精度トレーニング
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                vae_output, mu, logvar = vae_model(bert_embeddings)
                
                # 損失計算
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_div_per_batch = kl_div / bert_embeddings.size(0)  # バッチあたりのKL divergence
                kl_div = torch.maximum(
                    kl_div_per_batch,
                    torch.tensor(config.crop_lambda, device=kl_div.device, dtype=kl_div.dtype)
                )
                
                recon_loss = gemma_model.forward_teacher_forcing(
                    vae_output, 
                    batch_texts,
                    max_seq_len=config.max_seq_len
                )
                
                loss = recon_loss + beta * kl_div
            
            # 勾配蓄積
            #scaler.scale(loss).backward()
            loss.backward()

            
            if (step + 1) % config.grad_accum_steps == 0:
                optimizer.step()
            
            # ロギング
            writer.add_scalar("損失/訓練", loss.item(), epoch*len(train_loader)+step)
            writer.add_scalar("損失/再構成", recon_loss.item(), epoch*len(train_loader)+step)
            writer.add_scalar("損失/KL", kl_div.item(), epoch*len(train_loader)+step)
            writer.add_scalar("Beta", beta, epoch*len(train_loader)+step)
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_div.item():.4f}",
                "beta": f"{beta:.4f}"
            })
            optimizer.eval()  # 評価モード

            # サンプル生成
            if step % config.sample_interval == 0:
                generate_and_log_samples(
                    vae_model=vae_model,
                    bert_model=bert_model,
                    gemma_model=gemma_model,
                    device=device,
                    writer=writer,
                    global_step=epoch*len(train_loader)+step,
                    config=config
                )
            if step % config.ckpt_interval == 0:
                checkpoint = {
                    "epoch": epoch-1,
                    "model_state": vae_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss.item(),
                    "beta": beta,
                    "settings": config.__dict__
                }
                torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{datetime.datetime.now()}_{step}.pt"))
                torch.save(vae_model.state_dict(), os.path.join(config.checkpoint_dir, "latest_model.pt"))
            optimizer.train()
            # Beta更新
            beta = min(config.beta_max, beta + config.beta_step)
        
        # エポック終了時の処理
        generate_and_log_samples(
            vae_model=vae_model,
            bert_model=bert_model,
            gemma_model=gemma_model,
            device=device,
            writer=writer,
            global_step=(epoch+1)*len(train_loader),
            config=config
        )
        
        # チェックポイント保存
        checkpoint = {
            "epoch": epoch,
            "model_state": vae_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss.item(),
            "beta": beta,
            "settings": config.__dict__
        }
        
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}_{datetime.datetime.now()}.pt"))
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(vae_model.state_dict(), os.path.join(config.checkpoint_dir, f"best_model{datetime.datetime.now()}.pt"))

if __name__ == "__main__":
    disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="再開するチェックポイントのパス")
    args = parser.parse_args()
    
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル初期化
    bert_model, gemma_model, vae_model = initialize_models(config, device)
    optimizer = RAdamScheduleFree(vae_model.parameters(), lr=config.lr)
    
    # チェックポイントから再開
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        vae_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        beta = checkpoint["beta"]
        if "settings" in checkpoint:
            config.__dict__.update(checkpoint["settings"])
        logging.info(f"エポック {start_epoch} からトレーニングを再開")
    else:
        start_epoch = 0
        beta = config.beta_init
    
    # トレーニング開始
    train(
        config=config,
        vae_model=vae_model,
        bert_model=bert_model,
        gemma_model=gemma_model,
        optimizer=optimizer,
        device=device,
        start_epoch=start_epoch,
        initial_beta=beta
    )