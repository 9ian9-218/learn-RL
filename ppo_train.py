"""
PPO (Proximal Policy Optimization) 训练脚本 —— 用于大语言模型 RLHF 微调

================================================================================
PPO 整体算法流程（本脚本对应关系）
================================================================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  阶段 1：Rollout（采样 / 收集轨迹）                                       │
  │  generate_samples()                                                     │
  │    给定 prompt → Actor 自回归生成 response → 得到 (prompt+response) 序列   │
  └─────────────────────────────────────────────────────────────────────────┘
                                    ↓
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  阶段 2：评估轨迹（计算 PPO 所需各项量）                                    │
  │  generate_experiences()                                                 │
  │    ① old_log_probs  : 当前策略 π_θ 对生成 token 的对数概率（采样时的策略）  │
  │    ② ref_log_probs  : 参考策略 π_ref 的对数概率（用于 KL 惩罚，防偏离太远）│
  │    ③ values         : Critic 估计每个 token 位置的状态价值 V(s_t)         │
  │    ④ rewards        : 奖励模型分数 + 逐步 KL 惩罚 → 逐步 reward            │
  │    ⑤ advantages     : GAE 广义优势估计 A_t                               │
  │    ⑥ returns        : 目标回报 G_t = A_t + V(s_t)                        │
  └─────────────────────────────────────────────────────────────────────────┘
                                    ↓
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  阶段 3：PPO 策略更新（多 epoch 小批量梯度下降）                             │
  │  train_step()                                                           │
  │    策略损失: L^CLIP = -min(r_t·A_t, clip(r_t,1±ε)·A_t)  （裁剪重要性采样比）│
  │    价值损失: L^VF   = (V_θ(s_t) - G_t)²                 （拟合回报）       │
  │    分别更新 Actor 和 Critic 参数                                          │
  └─────────────────────────────────────────────────────────────────────────┘

  其中 r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log_prob - old_log_prob)
================================================================================
"""

# ---------- 依赖库 ----------
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass          # 用于定义轻量级数据容器（类似 struct）
from typing import Optional, Union, Tuple  # 类型注解，提高代码可读性
import random
import torch
import torch.nn.functional as F  # 常用函数：log_softmax、softmax 等
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import swanlab


# =============================================================================
# 数据集：将原始 prompt 文本转换为模型可接受的输入格式
# =============================================================================
class PromptDataset(Dataset):
    """
    提示词数据集。
    在 RLHF/PPO 中，prompt 是环境的"初始状态"，模型需要针对每个 prompt 生成 response。
    """

    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts          # 原始字符串列表
        self.tokenizer = tokenizer        # 与 Actor 模型配套的分词器

        self.final_prompts = []           # 格式化后的 prompt（可直接送入 generate）

        for prompt in prompts:
            if apply_chat_template:
                # 对话模型需要套 chat 模板，例如 <|user|>...<|assistant|>
                # add_generation_prompt=True 会在末尾追加 assistant 起始标记，引导模型开始生成
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(
                    content, tokenize=False, add_generation_prompt=True
                )
            else:
                # 非对话模型：在开头加 BOS（Beginning Of Sequence）标记
                prompt = self.tokenizer.bos_token + prompt

            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        # DataLoader 每次取一条格式化后的 prompt 字符串
        return self.final_prompts[index]


# =============================================================================
# Critic（价值网络 / 评论家）
# 作用：估计状态价值 V(s)，用于计算优势 A 和 baseline，降低策略梯度方差
# =============================================================================
class Critic(nn.Module):
    """
    价值模型：在 Actor 的 Transformer backbone 上加一个线性回归头。
    输出每个 token 位置的标量价值 V(s_t)。

    初始化方式：复用 Actor 的 base_model 权重（常见做法，共享表征）。
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model       # 共享 Actor 的 Transformer 主干
        self.base_model.eval()             # rollout 阶段 Critic 只做推理，不更新 backbone 的 BN/Dropout 行为
        # 回归头：hidden_size → 1，每个位置输出一个标量价值
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        self.value_head.to(dtype=next(base_model.parameters()).dtype)

    def forward(self, input_ids, attention_mask, num_actions):
        """
        参数:
            input_ids      : (batch, seq_len) 完整序列 token id
            attention_mask : (batch, seq_len) 1=有效 token，0=padding
            num_actions    : response 部分的 token 数（即"动作"数量）

        返回:
            values : (batch, num_actions) 每个生成 token 对应位置的价值估计
        """
        # 取最后一层 hidden state: (batch, seq_len, hidden_size)
        hidden_state = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        # 线性投影: (batch, seq_len, 1)
        value_model_output = self.value_head(hidden_state)
        # squeeze 最后一维 → (batch, seq_len)
        # [:, :-1] 去掉最后一个位置（没有对应的"下一 token"动作）
        # [:, -num_actions:] 只保留 response 部分的价值（与 action 对齐）
        values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
        return values


# =============================================================================
# PPO 核心损失函数
# =============================================================================

def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    """
    PPO-Clip 策略损失（要最小化，所以前面加负号）。

    公式:
        r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        L^CLIP = -E[ min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t) ]

    直觉:
        - A_t > 0（好动作）→ 希望增大 r_t（提高该动作概率），但 clip 限制更新幅度
        - A_t < 0（差动作）→ 希望减小 r_t，同样受 clip 保护
        - clip 防止策略一次更新走太远，保证"近端"优化

    参数:
        log_probs     : 当前策略下各 action 的对数概率 (batch, num_actions)
        old_log_probs : 采样时旧策略的对数概率 (batch, num_actions)
        advantages    : GAE 优势估计 (batch, num_actions)
        action_mask   : 有效 action 掩码，padding 位置不参与 loss 计算
        clip_eps      : 裁剪范围 ε，默认 0.2 → ratio 限制在 [0.8, 1.2]
    """
    # 重要性采样比 r_t = exp(log π_new - log π_old)
    ratio = (log_probs - old_log_probs).exp()
    # 未裁剪项：r_t * A_t
    surr1 = ratio * advantages
    # 裁剪项：clip(r_t, 1-ε, 1+ε) * A_t
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    # 取 min 后加负号 → 梯度下降时最大化 clipped surrogate objective
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()  # 对所有 token、所有样本求均值
    # 带 mask：只对有效 action 求平均（忽略 padding）
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    """
    价值函数损失：让 Critic 预测值接近实际回报 Gloss = -torch.min(surr1, surr2)_t。

    基础形式（MSE）:
        L^VF = (V_θ(s_t) - G_t)²

    可选 PPO 价值裁剪（与策略 clip 类似，防止价值网络更新过大）:
        V_clipped = V_old + clip(V_new - V_old, -ε, ε)
        L^VF = max( (V_clipped - G)², (V_new - G)² )

    参数:
        values     : 当前 Critic 预测 (batch, num_actions)
        old_values : 采样时的旧价值估计 (batch, num_actions)
        returns    : 目标回报 G_t = A_t + V(s_t) (batch, num_actions)
    """
    if clip_eps is not None:
        # 价值裁剪分支（本脚本 train_step 中 clip_eps=None，走 else 简单 MSE）
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        # 标准均方误差
        loss = (values - returns) ** 2

    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


# =============================================================================
# 经验回放缓冲区：暂存 rollout 收集的轨迹，供后续多 epoch 训练采样
# =============================================================================
class ExperienceBuffer:
    """
    存储 Experience 字典列表。
    PPO 是 on-policy 算法，通常每轮 rollout 后训练完就清空（见 train() 末尾 buffer.clear()）。
    """

    def __init__(self, limit):
        self.limit = limit    # 缓冲区最大容量（超出则丢弃最旧数据）
        self.buffer = []      # 实际存储列表

    def append(self, experiences):
        """将 Experience dataclass 列表转为 dict 并追加到 buffer"""
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",              # 完整 token 序列 (prompt + response)
            "action_log_probs",  # 旧策略对数概率
            "values",            # Critic 价值估计
            "returns",           # 目标回报
            "advantages",        # 优势估计
            "attention_mask",    # 序列有效位掩码
            "action_mask",       # response 部分有效位掩码
            "num_actions",       # response token 数
        )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value

        self.buffer.extend(batch)
        # 超出容量时保留最新的 limit 条
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit:]

    def get_batches(self, batch_size):
        """
        从缓冲区中随机采样 batch_size 条经验，返回组成一个 batch。
        这里通过 random.sample 随机从 buffer 列表中采样 batch_size 个元素，实现打乱采样（多 epoch 时有利于充分利用数据）。
        """
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


# =============================================================================
# 数据结构：采样结果 & 完整经验
# =============================================================================

@dataclass
class Samples:
    """
    Rollout 阶段的原始采样结果（尚未计算 reward/advantage）。
    每个 token 的"动作" = 在该位置选择的下一个 token id。
    """
    seqs: torch.Tensor                          # (batch, seq_len) prompt+response 完整序列
    attention_mask: Optional[torch.LongTensor]  # (batch, seq_len)
    action_mask: Optional[torch.BoolTensor]     # (batch, num_actions) response 有效位
    num_actions: Union[int, torch.Tensor]       # response 长度（动作数）
    packed_seq_lens: Optional[torch.Tensor]     # 预留：packed 格式序列长度
    response_length: torch.Tensor               # 每条样本实际 response 长度
    total_length: torch.Tensor                  # 每条样本实际总长度


@dataclass
class Experience:
    """
    完整经验 = Samples + PPO 训练所需的全部统计量。
    这是 PPO update 阶段的输入。
    """
    seqs: torch.Tensor
    action_log_probs: torch.Tensor   # π_old 下的 log prob（采样时冻结，用于 ratio 分母）
    values: torch.Tensor             # V(s_t)
    returns: Optional[torch.Tensor]  # G_t = A_t + V(s_t)
    advantages: Optional[torch.Tensor]  # A_t（GAE 估计）
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor             # 奖励模型原始输出（序列级）
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None  # 逐步 KL 散度估计


def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
):
    """
    近似 KL 散度（k1 估计）: KL(π_θ || π_ref) ≈ log π_θ - log π_ref

    在 RLHF 中，KL 惩罚防止微调后的策略偏离 SFT 参考模型太远，
    避免"奖励黑客"（reward hacking）导致输出退化。

    参数:
        log_probs: (batch_size, num_actions) 张量。
            每个元素 log_probs[i, j] 表示第 i 个样本第 j 个 token（动作位置）下，当前策略生成该 token 的 log 概率。
        ref_log_probs: (batch_size, num_actions) 张量。
            每个元素 ref_log_probs[i, j] 表示第 i 个样本第 j 个 token 下，参考策略生成该 token 的 log 概率。
        action_mask: (batch_size, num_actions) 张量（可选）。
            指示每个位置是否为有效动作（True/1）；padding 位置为 0。

    返回:
        log_ratio: (batch_size, num_actions) 张量。
            每个有效动作位置的对数概率差值（即近似 KL）。
    """
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask  # padding 位置 KL 置零

    return log_ratio


def get_advantages_and_returns(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    lambd: float,
):
    """
    用 GAE (Generalized Advantage Estimation) 计算优势 A_t 和回报 G_t。

    ── TD 误差 ──
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

    ── GAE 递推（从最后一个 token 往前扫）──
        A_t = δ_t + (γλ)·A_{t+1}
        边界条件: A_T = 0, V(s_{T+1}) = 0

    ── 回报 ──
        G_t = A_t + V(s_t)
        （PPO 中 Critic 的学习目标就是拟合 G_t）

    参数:
        gamma : 折扣因子 γ，未来奖励的衰减（本脚本 0.1，偏短视）
        lambd : GAE 参数 λ，偏差-方差权衡（0=纯 TD，1=蒙特卡洛）
    """
    lastgaelam = 0                    # A_{t+1}，初始为 0（episode 结束）
    advantages_reversed = []          # 倒序收集 A_t
    response_length = rewards.size(1) # = num_actions

    if action_mask is not None:
        # 将 padding 位置的 value/reward 置零，避免污染 GAE 计算
        values = action_mask * values
        rewards = action_mask * rewards

    # 从最后一个 action 位置向前递推
    for t in reversed(range(response_length)):
        # V(s_{t+1})：最后一个位置的下态价值视为 0
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        # TD 误差 δ_t
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        # GAE: A_t = δ_t + γλ·A_{t+1}
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)

    # 反转回正序 → (batch, num_actions)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # G_t = A_t + V(s_t)
    returns = advantages + values
    # detach：优势/回报作为常数目标，不参与 Critic 计算图的反向传播
    return advantages.detach(), returns


# =============================================================================
# 阶段 1：Rollout —— Actor 自回归生成 response
# =============================================================================

def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    """
    对每个 prompt 采样 n_samples_per_prompt 条 response，构成 on-policy 轨迹。

    流程:
        prompt → tokenize → model.generate() → 完整序列
        分离出 response 部分 → 构建 action_mask

    返回:
        samples_list: 每个 micro-batch 一个 Samples 对象
    """
    samples_list = []
    model.eval()  # 生成阶段关闭 dropout 等随机性（采样随机性来自 generate 的 temperature/top-p 等）

    # 每个 prompt 复制 n_samples_per_prompt 份，增加样本多样性
    # 例: [p1, p2] × 2 → [p1, p1, p2, p2]
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])

    # 分 micro-batch 生成，控制显存峰值
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        prompts = all_prompts[i : i + micro_rollout_batch_size]
        # 左填充到 max_length（对话模型常用左填充，保证生成对齐）
        inputs = actor_tokenizer(
            prompts, padding='max_length', max_length=max_length,
            truncation=True, return_tensors='pt'
        )
        input_ids = inputs['input_ids']  # (micro_batch, max_length)

        # 自回归生成：在 prompt 后续写 max_new_tokens 个 token
        seqs = model.generate(
            **inputs.to(device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,   # 遇到 EOS 提前停止
            pad_token_id=pad_token_id,
        )

        # 统一序列长度 = prompt 最大长度 + 最大生成长度
        target_len = max_new_tokens + max_length
        if seqs.size(1) >= target_len:
            seqs = seqs[:, :target_len]
        else:
            # 生成提前 EOS 结束导致长度不足 → 右侧 pad 补齐
            pad_width = target_len - seqs.size(1)
            seqs = torch.cat(
                [seqs, torch.full((seqs.size(0), pad_width), fill_value=pad_token_id, device=seqs.device)],
                dim=1,
            )

        # 全局 attention mask：非 pad 位置为 1
        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        # response 部分 = 去掉 prompt 前缀
        ans = seqs[:, input_ids.size(1):]
        # action_mask：response 中非 pad 的位置才是有效"动作"
        action_mask = ans.ne(pad_token_id).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),           # response 固定宽度
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),  # 每条样本实际生成长度
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    """
    构造逐步 reward 向量（仅最后有效 token 位置有奖励模型分数 + 全程 KL 惩罚）。

    RLHF 常见设计:
        - 奖励模型给出整个 response 的标量分数 → 放在最后一个 token
        - 每个 token 位置附加 -β·KL(π_θ||π_ref) 作为 shaping reward

    参数:
        kl               : 逐步 KL 估计 (batch, num_actions)
        r                : 奖励模型输出 (batch, 1) 序列级分数
        kl_ctl           : KL 惩罚系数 β
        clip_reward_value: 对奖励模型分数做 clip，防极端值
    """
    # 逐步 KL 惩罚 reward（每个 token 位置都有）
    # kl 的形状: (batch, num_actions)
    # kl_ctl 的形状: 标量或 (1,)
    kl_divergence_estimate = -kl_ctl * kl            # 形状: (batch, num_actions)
    rewards = kl_divergence_estimate                 # 形状: (batch, num_actions)

    # 每条样本 response 的有效长度（最后一个非 pad 位置索引 +1）
    ends = action_mask.sum(1)# 形状: (batch,1)

    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)

    # 裁剪奖励模型分数到 [-clip, +clip]
    reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)

    batch_size = r.size(0)
    for j in range(batch_size):
        # 将序列级奖励加到最后一个有效 action 位置
        # 只有最后一个 token 获得奖励信号，后续 GAE 算法会将该信号向前分配
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]

    # rewards 形状: (batch, num_actions)
    # 所有 action token 位置均包含 KL 散度项，仅最后一个 action token 累加 clipped r 奖励:
    #   - 非最后位置: reward = -β·KL
    #   - 最后一个位置: reward = -β·KL + clip(r)
    # 这种设计符合 RLHF 实践：reward model 分数仅加到 response 最后一个 token，其余位置只有 shaping KL 惩罚。
    return rewards


# =============================================================================
# 阶段 2：评估轨迹 —— 计算 log prob / value / reward / advantage
# =============================================================================

def generate_experiences(samples_list):
    """
    对 rollout 样本做全面评估，打包为 PPO 训练所需的 Experience。

    此阶段所有模型处于 eval + torch.no_grad()，只收集统计量，不更新参数。
    """
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []

    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
    

        with torch.no_grad():
            # ── ① 当前 Actor 策略的 log prob（即 π_old，后续更新时作为分母）──
            output = actor_model(seqs, attention_mask=attention_mask)
            logits = output.logits  # (batch, seq_len, vocab_size)
            # 因果 LM：位置 t 的 logits 预测 token t+1
            # 下面几步是在计算当前 actor 策略下，每个生成 token（动作）的对数概率（log prob）：
            # 1. F.log_softmax：对 logits（未归一化得分）做 softmax 归一化，并取对数，得到每个位置下每个 token 的 log 概率。
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq_len-1, vocab_size)
            # 2. gather：拿出实际生成的 token 的 log prob。
            #    seqs[:, 1:] 是目标 token 序列（第一个 token 不预测自己），unsqueeze(-1) 增加一个维度用于 gather 操作。
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            # 3. 由于一个序列 = prompt + response，我们只保留 response 部分的对数概率（即最后 num_actions 个）。
            #    squeeze(-1) 去掉最后一维（只剩 (batch, seq_len-1)），[:, -num_actions:] 截取 response 部分。
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

            # ── ② 参考模型 log prob（冻结的 SFT 模型，用于 KL 惩罚）──
            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

            # ── ③ Critic 价值估计 V(s_t) ──
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)

            # ── ④ 奖励模型：对整个 response 文本打分 ──
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            # 输出 shape (batch, 1)：序列级 reward（Outcome Reward Model）
            r = reward_model(**reward_model_inputs.to(device)).logits  # (batch_size, 1)
   

            # ── ⑤ 逐步 KL 散度 ──
            kl = compute_approx_kl(
                action_log_probs,
                ref_action_log_probs,
                action_mask=action_mask,
            ).to(device)

            # ── ⑥ 合成逐步 reward（KL 惩罚 + 末尾 RM 分数）──
            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2)

            # ── ⑦ GAE 计算优势和回报 ──
            advantages, returns = get_advantages_and_returns(
                value, rewards, action_mask, gamma=0.1, lambd=0.2
            )

        experiences.append(
            Experience(
                seqs,
                action_log_probs.detach(),  # 必须 detach：old_log_probs 是常数锚点
                value.detach(),
                returns.detach(),
                advantages.detach(),
                attention_mask,
                action_mask,
                r.detach(),
                samples.response_length,
                samples.total_length,
                num_actions,
                kl.detach(),
            )
        )

    return experiences


# =============================================================================
# 阶段 3 辅助：DataLoader 的 batch 拼接
# =============================================================================

@dataclass
class BufferItem:
    """collate_fn 输出的 batch 结构，供 train_step 消费"""
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


def collate_fn(batch):
    """将 buffer 中多条 experience dict 沿 batch 维拼接"""
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []

    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)

    return BufferItem(
        seqs, action_log_probs, values, returns, advantages,
        attention_mask, action_mask, action_mask.size(1),
    )


# =============================================================================
# 阶段 3：PPO 参数更新 —— 分别优化 Actor 和 Critic
# =============================================================================

def train_step(experience, steps):
    """
    单步 PPO 更新：
        1. 用当前 Actor 重新前向 → 得到新 log_probs
        2. 计算 clipped policy loss → 更新 Actor
        3. 用当前 Critic 重新前向 → 得到新 values
        4. 计算 value MSE loss → 更新 Critic

    注意：Actor 和 Critic 分开 backward/step，是简化实现；
          工业级框架（如 TRL）通常还会加 entropy bonus、梯度裁剪、KL early stopping 等。
    """
    # ── 更新 Actor（策略网络）──
    actor_model.train()
    optimizer_actor.zero_grad()

    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs  # 采样时冻结的 π_old
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns

    # 当前策略 π_θ 前向
    logits = actor_model(sequences, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    # PPO-Clip 策略损失
    policy_loss = compute_policy_loss(
        action_log_probs, old_action_log_probs, advantages, action_mask=action_mask
    )
    policy_loss.backward()
    optimizer_actor.step()
    # writer.add_scalar("policy_loss", policy_loss.item(), steps)

    # ── 更新 Critic（价值网络）──
    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    # writer.add_scalar("value_loss", value_loss.item(), steps)
    swanlab.log({
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }, step=steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")


# =============================================================================
# 主训练循环
# =============================================================================

def train():
    """
    外层循环结构:

        for episode in episodes:                    # 大轮次
            for prompts in dataloader:                # 取一批 prompt
                ① generate_samples   → rollout
                ② generate_experiences → 评估
                ③ buffer.append      → 存入经验池
                ④ for epoch in max_epochs:           # PPO 多 epoch 复用同一批数据
                       for batch in dataloader:
                           train_step()              # 更新 Actor + Critic
                ⑤ buffer.clear()     → on-policy：清空旧数据
    """
    buffer = ExperienceBuffer(limit=100)
    steps = 0

    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            # ① Rollout：Actor 生成 response
            samples = generate_samples(
                rand_prompts, actor_model, max_length, max_new_tokens,
                n_samples_per_prompt, micro_rollout_batch_size,
            )
            # ② 评估：计算 reward / advantage / return 等
            experiences = generate_experiences(samples)
            # ③ 存入经验池
            buffer.append(experiences)

            # ④ 用同一批经验做多 epoch 小批量梯度更新（PPO 的核心特征之一）
            dataloader = DataLoader(
                buffer, batch_size=micro_train_batch_size,
                shuffle=True, collate_fn=collate_fn,
            )
            torch.cuda.empty_cache()

            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1

            # ⑤ on-policy 清空：下一轮 rollout 必须基于更新后的策略重新采样
            buffer.clear()
            torch.cuda.empty_cache()


# =============================================================================
# 入口：模型 / 优化器 / 超参数初始化
# =============================================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 训练超参数 ──
    episodes = 3                  # 外层大循环轮数
    max_epochs = 5                # 每批经验重复训练 5 遍（PPO epoch）
    rollout_batch_size = 8        # 每次从 prompt 数据集取 8 条
    micro_rollout_batch_size = 2  # 生成时分 micro-batch=2，降低显存
    n_samples_per_prompt = 2      # 每个 prompt 采样 2 条 response（增加数据量）
    max_new_tokens = 50           # 最多生成 50 个 token（= 最大动作步数）
    max_length = 256              # prompt 最大 token 长度
    micro_train_batch_size = 2    # 训练时 micro-batch=2
    from torch.utils.tensorboard import SummaryWriter  # 训练曲线可视化
    # writer = SummaryWriter('./runs')  # TensorBoard 日志目录
    swanlab.init(
        project="learn-RL-ppo",
        experiment_name="ppo-qwen2.5-0.5b",
        logdir="./swanlog",
    )
    swanlab.config.update({
        "episodes": episodes,
        "max_epochs": max_epochs,
        "rollout_batch_size": rollout_batch_size,
        "micro_rollout_batch_size": micro_rollout_batch_size,
        "n_samples_per_prompt": n_samples_per_prompt,
        "max_new_tokens": max_new_tokens,
        "max_length": max_length,
        "micro_train_batch_size": micro_train_batch_size,
    })

    # ── 四个模型 ──
    # Actor：要被 RL 微调的策略模型（生成 response）
    actor_model = AutoModelForCausalLM.from_pretrained(
        '/home/z9ian9/downloads/models/Qwen/Qwen2.5-0.5B-Instruct'
    ).to(device)
    # Ref：冻结的参考模型（SFT 副本），仅用于 KL 惩罚，参数不更新
    ref_model = AutoModelForCausalLM.from_pretrained(
        '/home/z9ian9/downloads/models/Qwen/Qwen2.5-0.5B-Instruct'
    ).to(device)
    # Reward Model：人类偏好训练的打分模型，评估 response 质量
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        '/home/z9ian9/downloads/models/OpenAssistant/reward-model-deberta-v3-large-v2'
    ).to(device)

    actor_tokenizer = AutoTokenizer.from_pretrained('/home/z9ian9/downloads/models/Qwen/Qwen2.5-0.5B-Instruct')
    reward_tokenizer = AutoTokenizer.from_pretrained('/home/z9ian9/downloads/models/OpenAssistant/reward-model-deberta-v3-large-v2')

    # Critic：价值网络，复用 Actor backbone + 线性头
    critic_model = Critic(actor_model.base_model).to(device)

    # ── 优化器（Actor 和 Critic 分开优化）──
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)

    # 左填充：保证 batch 生成时 prompt 右对齐，response 从同一列开始
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id

    # ── 提示词数据集 ──
    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)

    train()
    swanlab.finish()
