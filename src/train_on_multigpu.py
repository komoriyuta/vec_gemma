# train.py
import os
import argparse
import logging
import random
from pathlib import Path
import datetime
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()
from datasets import disable_progress_bars
disable_progress_bars()
from datasets import load_dataset

from schedulefree import RAdamScheduleFree
from sentence_transformers import SentenceTransformer

from gemma.config import GemmaConfig, get_model_config
from gemma.model import GemmaForCausalLM
from VAEs.LinearVAE import LinearVAE
from VAEs.VAE import VAE
from VAEs.Perceptron import Perceptron

# ここでは、もともと torch.bfloat16 を使っていた部分を、指定精度（float16/float32）に変更します
torch.set_float32_matmul_precision('high')


class TrainingConfig:
    def __init__(self):
        # データ設定
        self.batch_size = 16
        self.max_seq_len = 256
        self.num_workers = 4

        # 最適化設定
        self.lr = 1e-4
        self.num_epochs = 10
        self.grad_accum_steps = 4
        self.beta_init = 0.1
        self.beta_max = 0.4
        self.beta_step = 1e-5
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
        self.log_dir = f"./logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
        self.checkpoint_dir = "./checkpoints"
        self.dataset_path = "AhmedSSabir/Japanese-wiki-dump-sentence-dataset"

        # 追加：数値精度の設定（"float16" または "float32"）
        self.precision = "float32"  

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


def prepare_dataset(config, epoch=0):
    def collate_fn(batch):
        return [item['text'] for item in batch]

    # データセットの読み込み（Arrow形式の場合は with_format("torch")）
    dataset = load_dataset(config.dataset_path, cache_dir="./.datasets")
    ds = dataset['train'].with_format("torch")
    
    # DDP対応：分散学習時は DistributedSampler を利用する
    if dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(ds, shuffle=True, seed=epoch)
        train_loader = DataLoader(
            ds,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    return train_loader


def initialize_models(config, device):
    # 使用する精度を選択：float16 または float32
    dtype = torch.float16 if config.precision == "float16" else torch.float32

    # BERTモデル
    bert_model = SentenceTransformer(config.bert_model_name)
    bert_model.requires_grad_(False)
    bert_model = bert_model.to(device)

    # Gemmaモデル
    from huggingface_hub import snapshot_download
    snapshot_dir = snapshot_download(repo_id='google/gemma-2-2b-jpn-it-pytorch')

    # tokenizer, checkpoint の存在確認
    tokenizer_path = os.path.join(snapshot_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'
    ckpt_path = os.path.join(snapshot_dir, f'model.ckpt')
    assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

    gemma_model_config = get_model_config("2b-v2")
    gemma_model_config.tokenizer = tokenizer_path

    # モデル構築と重み読み込み
    # ※ 必要に応じて、gemma_model_config の dtype を変更するなどの処理を追加してください
    torch.set_default_dtype(gemma_model_config.get_dtype())
    gemma_model = GemmaForCausalLM(gemma_model_config)
    gemma_model.requires_grad_(False)
    gemma_model.load_weights(ckpt_path)
    gemma_model = gemma_model.to(device).eval()

    # VAEモデル（学習対象）
    vae_model = LinearVAE(
        bert_model.get_sentence_embedding_dimension(),
        gemma_model.config.hidden_size,
        hidden_dim=config.vae_hidden_dim,
        latent_dim=config.vae_latent_dim
    ).to(device)

    return bert_model, gemma_model, vae_model


def generate_and_log_samples(vae_model, bert_model, gemma_model, device, writer, global_step, config):
    # メインプロセス以外はサンプル生成を行わない
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # 使用する精度
    dtype = torch.float16 if config.precision == "float16" else torch.float32

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
        # BERTで埋め込み取得（指定精度に変換）
        bert_embeddings = bert_model.encode(
            prep_texts,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        ).to(dtype)

        # VAEで埋め込み変換
        vae_output, _, _ = vae_model(bert_embeddings)

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
                    top_k=config.generation_top_k,
                    instructions=("<start_of_turn>user以下の内容をそのまま再現してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
                    embedding_length=30
                )
                generated_texts.append(generated[0])
            except Exception as e:
                generated_texts.append(f"生成エラー: {str(e)}")

        log_text = "\n\n".join([
            f"入力: {sample_texts[i]}\n生成結果: {generated_texts[i]}"
            for i in range(len(sample_texts))
        ])

        writer.add_text("生成サンプル", log_text, global_step)
        logging.info(f"\n=== ステップ {global_step} の生成サンプル ===\n{log_text}\n")

    vae_model.train()


def train(config, vae_model, bert_model, gemma_model, optimizer, device, start_epoch=0, initial_beta=0.1):
    # メインプロセスのみTensorBoard等のログ出力を行う
    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
    writer = SummaryWriter(config.log_dir) if is_main_process else None
    if is_main_process:
        setup_logging(config.log_dir)

    # プロンプトのリスト（前半, 後半）
    instructions = [
        # 基本パターン（指示あり）
        ("<start_of_turn>user次の文をそのまま出力して:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user以下の文を変更せずに出力してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>userこの文章をそのまま返してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user次の内容をそのまま出力してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user以下のテキストをそのまま返してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user次のデータを変更せずに出力してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>userこの情報をそのまま返してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user以下の内容をそのまま再現してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user次の文をそのまま表示してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>userこのデータを変更せずに返してください:\"", "\"\n<end_of_turn><start_of_turn>model\n"),

        # インストラクション付き
        ("<start_of_turn>user次の文をそのまま出力して:\"", "\"\nこの内容をそのまま出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user以下の文をそのまま返してください:\"", "\"\n意味を変えずにそのまま出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user次の内容を正確に再現してください:\"", "\"\nそのままの形で出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user次のデータを変更せずに出力してください:\"", "\"\n情報を保持したまま出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>userこの情報をそのまま返してください:\"", "\"\n入力されたまま出力してください。<end_of_turn><start_of_turn>model\n"),

        # "<start_of_turn>user\"" のみ
        ("<start_of_turn>user\"", "\"\n<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user\"", "\"\nそのまま出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user\"", "\"\nこの内容を変更せずに出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user\"", "\"\nこのままの形で出力してください。<end_of_turn><start_of_turn>model\n"),
        ("<start_of_turn>user\"", "\"\n入力通りにそのまま返してください。<end_of_turn><start_of_turn>model\n")
    ]

    best_loss = float('inf')
    beta = initial_beta
    optimizer.zero_grad()
    vae_model.train()

    # 使用する精度の設定
    dtype = torch.float16 if config.precision == "float16" else torch.float32

    for epoch in range(start_epoch, config.num_epochs):
        if is_main_process:
            logging.info(f"エポック {epoch+1}/{config.num_epochs}")
        train_loader = prepare_dataset(config, epoch=epoch)
        total_steps = len(train_loader)
        progress_bar = tqdm(
            train_loader,
            total=total_steps,
            desc=f"エポック {epoch+1}/{config.num_epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}",
            dynamic_ncols=True
        ) if is_main_process else train_loader

        for step, batch_texts in enumerate(progress_bar):
            # BERTで埋め込み取得（指定精度に変換）
            with torch.no_grad():
                rep_texts = ["文章: " + text for text in batch_texts]
                bert_embeddings = bert_model.encode(
                    rep_texts,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False
                ).to(dtype)

            # float16 の場合は autocast を利用、それ以外はそのまま実行
            context = torch.autocast(device_type='cuda', dtype=dtype) if dtype == torch.float16 else nullcontext()
            with context:
                vae_output, mu, logvar = vae_model(bert_embeddings)
                # KLダイバージェンスの計算
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_div_per_batch = kl_div / bert_embeddings.size(0)
                kl_div = torch.maximum(
                    kl_div_per_batch,
                    torch.tensor(config.crop_lambda, device=kl_div.device, dtype=kl_div.dtype)
                )

                recon_loss = gemma_model.forward_teacher_forcing(
                    vae_output,
                    batch_texts,
                    max_seq_len=config.max_seq_len,
                    instructions=instructions[random.randint(0, len(instructions)-1)],
                    embedding_length=30
                )
                loss = recon_loss + beta * kl_div

            loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # ログ出力（メインプロセスのみ）
            global_step = epoch * len(train_loader) + step
            if is_main_process and writer is not None:
                writer.add_scalar("損失/訓練", loss.item(), global_step)
                writer.add_scalar("損失/再構成", recon_loss.item(), global_step)
                writer.add_scalar("損失/KL", kl_div.item(), global_step)
                writer.add_scalar("Beta", beta, global_step)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_div.item():.4f}",
                    "beta": f"{beta:.4f}"
                })

            # サンプル生成（メインプロセスのみ）
            if (step % config.sample_interval == 0) and is_main_process:
                generate_and_log_samples(
                    vae_model=vae_model,
                    bert_model=bert_model,
                    gemma_model=gemma_model,
                    device=device,
                    writer=writer,
                    global_step=global_step,
                    config=config,
                )

            # チェックポイント保存（メインプロセスのみ）
            if (step % config.ckpt_interval == 0) and is_main_process:
                model_state = vae_model.module.state_dict() if hasattr(vae_model, "module") else vae_model.state_dict()
                checkpoint = {
                    "epoch": epoch - 1,
                    "model_state": model_state,
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss.item(),
                    "beta": beta,
                    "settings": config.__dict__
                }
                ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{step}.pt")
                torch.save(checkpoint, ckpt_path)
                torch.save(model_state, os.path.join(config.checkpoint_dir, "latest_model.pt"))

            beta = min(config.beta_max, beta + config.beta_step)

        # エポック終了時のサンプル生成（メインプロセスのみ）
        if is_main_process:
            generate_and_log_samples(
                vae_model=vae_model,
                bert_model=bert_model,
                gemma_model=gemma_model,
                device=device,
                writer=writer,
                global_step=(epoch+1) * len(train_loader),
                config=config
            )
            model_state = vae_model.module.state_dict() if hasattr(vae_model, "module") else vae_model.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "loss": loss.item(),
                "beta": beta,
                "settings": config.__dict__
            }
            ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
            torch.save(checkpoint, ckpt_path)

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model_state, os.path.join(config.checkpoint_dir, f"best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"))

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="再開するチェックポイントのパス")
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float32", help="学習精度（float16 または float32）")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP用のlocal_rank")
    args = parser.parse_args()

    config = TrainingConfig()
    # コマンドライン引数で指定された精度に更新
    config.precision = args.precision

    # DDP対応：local_rank 引数または環境変数からプロセスランクを取得
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル初期化
    bert_model, gemma_model, vae_model = initialize_models(config, device)

    # チェックポイントから再開する場合は、DDPラッピング前に読み込む
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        vae_model.load_state_dict(resume_checkpoint["model_state"])
    # DDP対応：学習対象のモデルのみラップ（BERT, Gemma は固定）
    if args.local_rank != -1:
        vae_model = torch.nn.parallel.DistributedDataParallel(vae_model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = RAdamScheduleFree(vae_model.parameters(), lr=config.lr)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        start_epoch = resume_checkpoint["epoch"] + 1
        beta = resume_checkpoint["beta"]
        if "settings" in resume_checkpoint:
            config.__dict__.update(resume_checkpoint["settings"])
        if dist.is_initialized() and dist.get_rank() == 0:
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

    # DDPプロセスの場合、終了前にグループを破棄
    if dist.is_initialized():
        dist.destroy_process_group()
