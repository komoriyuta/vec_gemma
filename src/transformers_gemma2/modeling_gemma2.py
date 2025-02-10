import torch
import torch.nn.functional as F
from transformers import Gemma2ForCausalLM, AutoTokenizer
from typing import List, Tuple, Union

class Gemma2ForAutoEncoding(Gemma2ForCausalLM):
    """
    Transformers版 Gemma2 を継承して、以下の特殊メソッドを追加したクラスです。
      - generate_with_initial_embedding(...)
          ・VAE などで得た「初期埋め込み」を dummy トークンの位置に注入し，
            instruction（条件）を与えた上で自己回帰生成を行うメソッド
      - encode_texts(...)
          ・テキストをトークン化し，固定長パディングと attention mask を作成するユーティリティ
      - forward_teacher_forcing(...)
          ・VAE 埋め込みと instruction を条件として与えた上で，ターゲット文に対する
            教師強制学習（Teacher Forcing）を行い，損失を計算するメソッド

    ※ 内部では self.model.embed_tokens や self.lm_head を利用しており，PEFT（例：LoRA）
       のためのパラメータ更新もそのまま行えます。
    """
    def __init__(self, config, tokenizer_name_or_path: str):
        # Gemma2ForCausalLM の初期化（from_pretrained で読み込む場合も同様）
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        # tokenizer の pad/eos トークンID を取得（必要に応じて CLS なども）
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        # Transformers 版 Gemma2 は Gemma2Model を self.model として持っている

    def encode_texts(
        self,
        texts: List[str],
        max_seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        テキストリストをトークン化し、最大長 max_seq_len にパディングしたテンソルと
        attention mask を返すユーティリティです。
        """
        token_ids = []
        for text in texts:
            # add_special_tokens=False として必要な場合のみ BOS/EOS を付与（BERT なら CLS/EOS など）
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if add_bos:
                # ここでは例として CLS トークンを BOS としています（必要に応じて変更）
                tokens = [self.tokenizer.cls_token_id] + tokens
            if add_eos:
                tokens = tokens + [self.eos_id]
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
                if add_eos:
                    tokens[-1] = self.eos_id
            padding = [self.pad_id] * (max_seq_len - len(tokens))
            token_ids.append(tokens + padding)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        attention_mask = (token_ids != self.pad_id).long()
        return token_ids, attention_mask

    @torch.no_grad()
    def generate_with_initial_embedding(
        self,
        initial_embedding: torch.Tensor,  # shape: [batch_size, 1, hidden_size]
        output_len: int = 100,
        temperature: float = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        instructions: Tuple[str, str] = ("", ""),  # (instruction_prompt0, instruction_prompt1)
    ) -> List[str]:
        """
        VAE 埋め込みと 2 つの instruction（条件）を与え、以下の構成の入力シーケンスから
        自己回帰生成を行います。

          入力シーケンス = [instruction_prompt0 のトークン列] + [dummy] + [instruction_prompt1 のトークン列] + [生成トークン...]

        dummy の位置は後で initial_embedding により上書きされます。
        """
        batch_size = initial_embedding.size(0)
        # トークン化（ここでは add_special_tokens=False でシンプルに）
        instruction_prompt0, instruction_prompt1 = instructions
        seq0 = self.tokenizer.encode(instruction_prompt0, add_special_tokens=False) if instruction_prompt0 else []
        seq1 = self.tokenizer.encode(instruction_prompt1, add_special_tokens=False) if instruction_prompt1 else []
        dummy_token = self.pad_id  # ダミートークンとして pad_id を一時使用
        prefix_tokens = seq0 + [dummy_token] + seq1
        prefix_len = len(prefix_tokens)
        total_seq_len = prefix_len + output_len

        # 入力トークン列を作成（すべて pad で初期化）
        generated = torch.full((batch_size, total_seq_len), self.pad_id, dtype=torch.long, device=initial_embedding.device)
        prefix_tensor = torch.tensor(prefix_tokens, dtype=torch.long, device=initial_embedding.device)
        generated[:, :prefix_len] = prefix_tensor.unsqueeze(0).expand(batch_size, -1)

        # Gemma2ForCausalLM では inputs_embeds を渡すことが可能なので，
        # embed_tokens を利用して prefix の埋め込みを取得
        normalizer = self.config.hidden_size ** 0.5
        prefix_embeds = self.model.embed_tokens(generated[:, :prefix_len])
        # dummy の位置（インデックス＝len(seq0)）の埋め込みを initial_embedding に置換
        dummy_index = len(seq0)
        prefix_embeds[:, dummy_index, :] = initial_embedding.squeeze(1)
        prefix_embeds = prefix_embeds * normalizer

        # prefix 部分から初期状態（past_key_values など）を取得
        outputs = self.model(
            inputs_embeds=prefix_embeds,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        cur_len = prefix_len
            # 自己回帰生成ループ
            # 生成ループ内（generate_with_initial_embedding の該当部分）：
        while cur_len < total_seq_len:
                # 直前のトークンの埋め込みを用意
                last_token_ids = generated[:, cur_len - 1].unsqueeze(1)
                last_embeds = self.model.embed_tokens(last_token_ids) * normalizer

                # 現在のシーケンス長（キャッシュに含まれるトークン数）を明示的に計算して位置情報を作成
                curr_seq_len = past_key_values.get_seq_length() if past_key_values is not None else prefix_embeds.size(1)
                position_ids = torch.arange(curr_seq_len, curr_seq_len + 1, device=last_embeds.device).unsqueeze(0)

                outputs = self.model(
                    inputs_embeds=last_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=position_ids,  # ここで明示的に位置情報を渡す
                )
                logits = self.lm_head(outputs[0])  # [B, 1, vocab_size]
                logits = logits[:, -1, :]  # [B, vocab_size]
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                # (ここで top-p / top-k サンプリングの処理)
                next_tokens = torch.multinomial(probs, num_samples=1)
                generated[:, cur_len] = next_tokens.squeeze(-1)
                cur_len += 1
                past_key_values = outputs.past_key_values

            # prefix 部分以降をデコード
        outputs_text = []
        for i in range(batch_size):
                gen_tokens = generated[i, prefix_len:].tolist()
                if self.eos_id in gen_tokens:
                    gen_tokens = gen_tokens[: gen_tokens.index(self.eos_id)]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                outputs_text.append(text)
        return outputs_text

    def forward_teacher_forcing(
        self,
        vae_embedding: torch.Tensor,  # shape: [batch_size, 1, hidden_size]
        target_texts: List[str],
        max_seq_len: int = 512,
        instructions: Tuple[str, str] = ("", ""),
    ) -> torch.Tensor:
        """
        教師強制（Teacher Forcing）を行うメソッドです。
        入力シーケンスは次の部分の連結となります：
           [instruction_prompt0 のトークン列] + [dummy] + [instruction_prompt1 のトークン列] + [target_text のトークン列]
        ここで dummy の位置は vae_embedding により上書きされ，損失は target_text 部分のみで計算します。
        """
        batch_size = vae_embedding.size(0)
        instruction_prompt0, instruction_prompt1 = instructions
        seq0 = self.tokenizer.encode(instruction_prompt0, add_special_tokens=False) if instruction_prompt0 else []
        seq1 = self.tokenizer.encode(instruction_prompt1, add_special_tokens=False) if instruction_prompt1 else []
        dummy_token = self.pad_id
        prefix_tokens = seq0 + [dummy_token] + seq1
        prefix_len = len(prefix_tokens)

        # 各サンプルの target_text を特殊トークン付きでエンコード
        target_ids_list = []
        for text in target_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > (max_seq_len - prefix_len):
                tokens = tokens[: (max_seq_len - prefix_len)]
                tokens[-1] = self.eos_id
            target_ids_list.append(tokens)

        final_tokens = []
        for tokens in target_ids_list:
            seq = prefix_tokens + tokens
            if len(seq) < max_seq_len:
                seq = seq + [self.pad_id] * (max_seq_len - len(seq))
            else:
                seq = seq[:max_seq_len]
            final_tokens.append(seq)
        final_tokens = torch.tensor(final_tokens, dtype=torch.long, device=vae_embedding.device)
        attention_mask = (final_tokens != self.pad_id).long()

        normalizer = self.config.hidden_size ** 0.5
        # 埋め込みを取得（inputs_embeds を用いるので input_ids は不要）
        token_embeddings = self.model.embed_tokens(final_tokens) * normalizer
        # dummy の位置（インデックス＝len(seq0)）を vae_embedding で上書き
        dummy_index = len(seq0)
        token_embeddings[:, dummy_index, :] = vae_embedding.squeeze(1) * normalizer

        outputs = self.model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = self.lm_head(outputs[0])  # [B, max_seq_len, vocab_size]
        # 損失は prefix 部分以降（target_text 部分）のみ計算
        shift_logits = logits[:, prefix_len - 1:-1, :].contiguous()
        shift_labels = final_tokens[:, prefix_len:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_id,
        )
        return loss
