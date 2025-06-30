# ref-adamw-mini-ScheduleFree
"Ref" stands for both "Refine" and "Reflex" , self-evolving, self-supervised, reinforcement learning

これは試験中のoptimizerです(意図せぬ動作をする場合があります)

#Ref-AdamW-mini-SF

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

## 謝辞(Acknowledgments)

本プロジェクトは、[@zyushun](https://github.com/zyushun) 氏による [Adam-mini](https://github.com/zyushun/Adam-mini) の素晴らしい先行研究と実装に多くを学び、その上に構築しています。軽量かつ高性能な最適化器の礎を築いていただき、深く感謝申し上げます。

また、PyTorch および OSS コミュニティの皆さま、Schedule-Free 最適化や mixed precision 学習に関する研究を築いてきた研究者の方々の知見に、心より敬意を表します。

さらに、本実装にあたっては GitHub Copilot との協働も大きな助けとなりました。AI支援による開発の可能性に感謝するとともに、これからも人間とAIの共創が広がることを願っています。