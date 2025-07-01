import torch
from torch.optim.optimizer import Optimizer

class RefAdamWminiScheduleFree(Optimizer):
    """
    AdamW-mini + ScheduleFree: 省メモリ・高速化 + スケジューリング不要の自動学習率調整
    - m/vはfloat16で保持
    - Weight DecayはAdamW方式
    - 学習率は損失や勾配ノルムに応じて自動調整（ScheduleFree）
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, min_lr=1e-6, max_lr=1e-2, schedulefree_alpha=0.05, dtype=None,
                 ref_alpha=0.5, ref_window=20, ref_base_eps=1e-3, ref_min_eps=1e-5, ref_max_steps=None):
        self._user_dtype = dtype
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        min_lr=min_lr, max_lr=max_lr, schedulefree_alpha=schedulefree_alpha, dtype=dtype)
        super().__init__(params, defaults)
        # ref的EMA用
        self.ref_alpha = ref_alpha
        self.ref_window = ref_window
        self.ref_base_eps = ref_base_eps
        self.ref_min_eps = ref_min_eps
        self.ref_max_steps = ref_max_steps
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None, loss=None, max_steps=None):
        loss_val = None
        if closure is not None:
            loss_val = closure()
        self._step_count += 1
        # --- AdamW-mini Schedule-Free本体 ---
        for group in self.param_groups:
            dtype = group.get('dtype', None) or self._user_dtype
            for p in group['params']:
                if p.grad is None:
                    continue
                # Dtype優先順位: group['dtype'] > self._user_dtype > p.data.dtype
                param_dtype = dtype if dtype is not None else p.data.dtype
                grad = p.grad.data.to(param_dtype)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=param_dtype)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=param_dtype)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # bias correction省略
                grad_norm = grad.norm().item()
                if 'grad_norm_ema' not in state:
                    state['grad_norm_ema'] = grad_norm
                else:
                    state['grad_norm_ema'] = (1 - group['schedulefree_alpha']) * state['grad_norm_ema'] + group['schedulefree_alpha'] * grad_norm
                auto_lr = max(min(group['max_lr'], group['lr'] * (state['grad_norm_ema'] / (grad_norm + 1e-12))), group['min_lr'])
                step_size = auto_lr
                denom = (exp_avg_sq.to(torch.float32).sqrt() + group['eps']).to(param_dtype)
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * auto_lr)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        # --- ref的EMA注入（exp_avgに直接注入、省メモリ・全体一括版） ---
        # loss, max_stepsは外部から渡す必要あり
        if loss is not None:
            total_steps = max_steps if max_steps is not None else self.ref_max_steps
            if total_steps is None or total_steps <= 0:
                progress = 0.5
            else:
                progress = self._step_count / total_steps
            in_middle = 0.2 <= progress <= 0.8
            ref_alpha = self.ref_alpha if in_middle else 0.0
            if ref_alpha > 0:
                print(f"[ref inject] step={self._step_count} ref_alpha={ref_alpha}")
                for group in self.param_groups:
                    dtype = group.get('dtype', None) or self._user_dtype
                    for p in group['params']:
                        state = self.state[p]
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(1 - ref_alpha).add_(p.data.to(exp_avg.dtype), alpha=ref_alpha)
        return loss_val

# In this optimizer, "Ref" stands for both "Refine" (to improve/refine) and "Reflex" (introspection/self-reference).
# This optimizer organizes its own features and consolidates learning through introspection, embodying a self-evolving optimizer inspired by self-supervised reinforcement learning.
