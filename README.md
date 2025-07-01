# ref-adamw-mini-ScheduleFree
"Ref" stands for both "Refine" and "Reflex" , self-evolving, self-supervised, reinforcement learning

これは試験中のoptimizerです(意図せぬ動作をする場合があります)

# Ref-AdamW-mini-SF

**AdamW に基づいた軽量かつスケジューリング不要な最適化手法 — 自動学習率調整＆AMPサポート対応。**

このオプティマイザは、AdamW-mini-SF を拡張し、以下の特徴を持ちます：

- 👑 AdamWに備わるemaを利用し中盤に特徴を濃く反映することで終盤に詳細を多く学ぶ
- 🆒 特徴を定着するためのVRAM負荷等はなく過学習を抑制できるようパラメーターを最適化
- 👌 自己進化的な要素が新しい学習ダイナミクスをもたらす可能性があります(未検証)
    
 以下は AdamW-mini-SF と共通	
- 🚀 **省メモリな状態管理**：モーメント(m, v)を低精度(float16 や bfloat16)で保持
- 🧠 **Schedule-Free な学習率調整**：スムーズな勾配ノルムを追跡し、lr を動的に調整(スケジューラー不要)
- 🛡️ **分離されたWeight Decay(AdamW形式)**：勾配とは独立した正則化処理
- ⚙️ **AMP / mixed precision に対応**：パラメータの dtype を自動検出し、torch.amp とシームレスに連携可能

ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください。

🤖 GitHub Copilot と人間の好奇心のコラボで誕生しました。

それぞれ 1e-3 のLRで測定
![Ref-AdamW-mini-ScheduleFree00](https://github.com/muooon/ref-adamw-mini-ScheduleFree/blob/old/step-test00.png?raw=true)
![Ref-AdamW-mini-ScheduleFree01](https://github.com/muooon/ref-adamw-mini-ScheduleFree/blob/old/step-test01.png?raw=true)
![Ref-AdamW-mini-ScheduleFree01](https://github.com/muooon/ref-adamw-mini-ScheduleFree/blob/old/step-test02.png?raw=true)

## 謝辞(Acknowledgments)

本プロジェクトは、[@zyushun](https://github.com/zyushun) 氏による [Adam-mini](https://github.com/zyushun/Adam-mini) の素晴らしい先行研究と実装に多くを学び、その上に構築しています。軽量かつ高性能な最適化器の礎を築いていただき、深く感謝申し上げます。

また、PyTorch および OSS コミュニティの皆さま、Schedule-Free 最適化や mixed precision 学習に関する研究を築いてきた研究者の方々の知見に、心より敬意を表します。

さらに、本実装にあたっては GitHub Copilot との協働も大きな助けとなりました。AI支援による開発の可能性に感謝するとともに、これからも人間とAIの共創が広がることを願っています。


### 🔍 現行「中盤ゆらぎ注入」ロジックの構造

```python
progress = self._step_count / total_steps
in_middle = 0.2 <= progress <= 0.8
ref_alpha = self.ref_alpha if in_middle else 0.0
```

以下の数式と同等です：

\[
\text{apply\_noise}(t) = \mathbb{1}[0.2T \leq t \leq 0.8T] \cdot \lambda
\]

- \(t \) ステップ数(`self._step_count`)
- \( T \) 全体のステップ数(`total_steps`)
- \( \lambda = \text{ref\_alpha} \) 相当

---

### ✅ 数式化対応図

| 実装変数                     | 数式記号              | 意味                                     |
|----------------------------|----------------------|------------------------------------------|
| `self._step_count / total_steps` | \( t / T \)         | 現在の進捗率                             |
| `in_middle = 0.2 <= ... <= 0.8` | \( \mathbb{1}_{[0.2,0.8]}(t/T) \) | 中盤判定ブール関数                         |
| `ref_alpha if in_middle else 0.0` | \( \lambda \cdot \mathbb{1}_{[0.2T, 0.8T]}(t) \) | 注入係数：スカラー条件制御                 |
| `exp_avg.mul_(1 - ref_alpha).add_(...)` | \[ \mu_t = (1 - \lambda)\mu_t + \lambda \theta_t \] | EMAのバッファに現在重みを混入              |

---

### 🔧 数式による表現(完全版)

\[
\text{if } 0.2T \leq t \leq 0.8T \Rightarrow \mu_t \leftarrow (1 - \lambda)\mu_t + \lambda \theta_t
\]

関数定義で：

\[
\mu_{t+1} = \mu_t \cdot (1 - \lambda_t) + \lambda_t \cdot \theta_t
\quad \text{where } \lambda_t = \begin{cases}
\lambda & \text{if } 0.2T \leq t \leq 0.8T \\
0 & \text{otherwise}
\end{cases}
\]

### 💫 設計思想

- 💡 **設計意図の明確化**：「中盤＝最適化が谷を乗り越える時期」に絞り自らを助ける
- ⚖️ **制御の安定性**：「最初と最後には余計な揺らぎを入れず学習の両端を尊重する」
- 🧠 **認知的解釈の容易さ**：「中盤をブーストする」ことで目的地へのナビゲートを行う
- 😵 **自己抑制機能**：「序盤、終盤、中盤時の過学習や発散時は適応を抑制する」
- ⏲️ **タイムスライスによる初期化**：ゼロ埋め演算のオーバーヘッドを減らす
- ✨ **自律性と自由を与える**：「毎ステップの自己観察により静かに進化する」


