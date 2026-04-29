# 🤖 可解釋性 DRL 動態導航平台
### Explainable Deep Reinforcement Learning Navigation Platform

> 透過 D3QN 演算法與 SHAP 解釋技術，實現動態避障導航並將 AI 的「思考過程」視覺化。

## 🎬 專案展示

<p align="center">
  <video src="DRL_FinalProject.mp4" width="720" controls>
    Your browser does not support the video tag.
  </video>
</p>

---

## 📋 目錄

- [專案展示](#-專案展示)
- [專案背景](#專案背景)
- [問題陳述](#問題陳述)
- [解決方案](#解決方案)
- [系統架構](#系統架構)
- [MDP 建模](#mdp-建模)
- [開發與實作](#開發與實作)
- [實驗設計](#實驗設計)
- [預期成果](#預期成果)
- [S.M.A.R.T. 目標](#smart-目標)
- [專案時程](#專案時程)
- [技術棧](#技術棧)
- [快速開始](#快速開始)

---

## 專案背景

在工業 4.0 的浪潮下，自動化倉儲（如 Amazon Robotics）已是標配。然而，當數百台機器人在動態變化的環境中穿梭時，傳統路徑演算法（如 A\*）往往因計算延時或無法應對突發障礙而導致「塞車」或碰撞，損失數百萬美元的運作效率。

## 問題陳述

現有的強化學習導航系統常被視為「**黑盒子**」。當機器人做出異常轉向或停滯時，工程師難以理解其內部的決策邏輯。此外，在離散的網格地圖中，如何精準定義 MDP 的狀態空間以平衡「**運算速度**」與「**導航精度**」，仍是學術與實務上的巨大挑戰。

## 解決方案

本專案提出一個「**可解釋性 DRL 導航平台**」。透過建立精確的 MDP 模型，並導入 **D3QN (Dueling Double Deep Q-Network)** 演算法，不僅能實現動態避障，更結合以下技術將 AI 的「思考過程」視覺化：

| 技術 | 說明 |
|------|------|
| **SHAP 解釋技術** | 量化各特徵對決策的貢獻度 |
| **策略熱圖 (Policy Heatmap)** | 視覺化機器人在每個位置的偏好動作 |
| **Q-Value 分佈圖** | 即時渲染各動作的估計價值 |

---

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                    前端視覺化介面                        │
│              (Python Dash / Streamlit)                  │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │  Q-Value    │ │  Policy     │ │  SHAP Feature     │  │
│  │  分佈圖     │ │  熱圖       │ │  權重圖           │   │
│  └─────────────┘ └─────────────┘ └───────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    後端訓練引擎                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           D3QN (Dueling Double DQN)             │    │
│  │         Stable Baselines3 Framework             │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│                    模擬環境                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │      OpenAI Gymnasium Custom Grid-World         │    │
│  │        (動態障礙物隨機移動)                       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## MDP 建模

### 狀態空間 $S$ (State)

包含以下資訊：

- **機器人座標** — 當前位置 $(x, y)$
- **目標向量** — 相對於目標的方向與距離
- **局部感測** — 周圍 3×3 網格的障礙物偵測資訊

### 動作空間 $A$ (Action)

| 動作 | 說明 |
|------|------|
| ⬆️ 前進 | 向當前朝向移動一格 |
| ⬇️ 後退 | 向反方向移動一格 |
| ⬅️ 左轉 | 向左移動一格 |
| ➡️ 右轉 | 向右移動一格 |
| ⏸️ 停止 | 原地等待 |

### 獎勵函數 $R$ (Reward)

$$R = R_{\text{goal}} + R_{\text{step}} + R_{\text{collision}}$$

| 獎勵成分 | 說明 |
|----------|------|
| $R_{\text{goal}}$ | 到達目標時給予正獎勵 |
| $R_{\text{step}}$ | 每步給予小額負獎勵，鼓勵效率 |
| $R_{\text{collision}}$ | 碰撞障礙物時給予大額負獎勵 |

> 精細調校各項權重以避免稀疏獎勵 (Sparse Reward) 問題。

---

## 開發與實作

### 環境模擬

使用 **OpenAI Gymnasium** 自定義網格環境 (Grid-World)，模擬障礙物隨機移動的動態場景。初期設定為 **10×10** 的網格規模。

### 模型訓練

採用 **D3QN (Dueling Double Deep Q-Network)** 架構，基於 **Stable Baselines3** 框架進行開發與訓練。

### 視覺化系統

使用 **Python Dash** 或 **Streamlit** 建立前端互動介面，支援：

- 即時渲染 Q-Value 分佈
- 策略熱圖 (Policy Heatmap)
- SHAP 特徵權重分析
- 注意力區塊視覺化

---

## 實驗設計

### 1. 基準測試 (Baseline Comparison)

將 DRL 模型與以下基準進行對比：

| 方法 | 說明 |
|------|------|
| **A\* 演算法** | 傳統最短路徑搜索 |
| **隨機策略 (Random Policy)** | 隨機選取動作作為下界參考 |
| **D3QN (Ours)** | 本專案提出的方法 |

### 2. 變數控制 — 魯棒性測試 (Robustness)

- 改變動態障礙物的 **數量**
- 改變動態障礙物的 **移動速度**
- 測試模型在不同環境複雜度下的表現

### 3. 消融實驗 (Ablation Study)

測試加入「視覺化回饋」前後，人類操作員排解系統異常所需的 **平均時間差異**。

---

## 預期成果

- ✅ DRL 模型在動態環境下的 **路徑成功率** 預計比 A\* 高出 **20% 以上**
- ✅ 透過視覺化平台，開發者能成功辨識出模型在哪些特定網格配置下會產生「**決策遲疑**」
- ✅ 針對性地優化獎勵函數，達成 **端到端的可解釋導航**

---

## S.M.A.R.T. 目標

| 維度 | 目標 |
|------|------|
| **Specific** (具體的) | 開發一個基於 DRL 的動態網格導航平台，包含後端 MDP 訓練邏輯與前端策略視覺化介面 |
| **Measurable** (可衡量的) | ① 導航成功率 ≥ **95%**　② 平均每步決策延遲 < **50ms**　③ 視覺化界面顯示 ≥ **3 種**解釋性指標 |
| **Achievable** (可達成的) | 利用 Stable Baselines3 與 Gymnasium 進行開發，初期 10×10 網格規模，技術路徑清晰 |
| **Relevant** (具相關性的) | 直接解決 AI 導航的「黑盒子」問題，對工業自動化與決策透明化具有高度實務價值 |
| **Time-bound** (有時限的) | 預計 **8 週**完成全部開發與實驗 |

---

## 專案時程

```
Week 1-2 ████████░░░░░░░░░░░░░░░░  環境搭建與 MDP 建模
Week 3-5 ░░░░░░░░████████████░░░░  模型訓練與優化
Week 6-8 ░░░░░░░░░░░░░░░░████████  介面開發與實驗分析
```

| 階段 | 時間 | 工作內容 |
|------|------|----------|
| **Phase 1** | 第 1–2 週 | 環境搭建、Gymnasium 自定義環境開發、MDP 建模 |
| **Phase 2** | 第 3–5 週 | D3QN 模型訓練、超參數調校、獎勵函數優化 |
| **Phase 3** | 第 6–8 週 | 前端介面開發、SHAP 整合、實驗分析與報告撰寫 |

---

## 技術棧

| 類別 | 技術 |
|------|------|
| **強化學習框架** | Stable Baselines3、PyTorch |
| **模擬環境** | OpenAI Gymnasium (Custom Grid-World) |
| **演算法** | D3QN (Dueling Double Deep Q-Network) |
| **可解釋性** | SHAP (SHapley Additive exPlanations) |
| **前端視覺化** | Python Dash / Streamlit |
| **程式語言** | Python 3.10+ |

---

## 快速開始

### 環境安裝

```bash
# 1. 複製專案
git clone https://github.com/kaiiicoding94/DRL_FinalProject.git
cd DRL_FinalProject

# 2. 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 3. 安裝依賴
pip install -r requirements.txt
```

### 訓練模型

```bash
python train.py --grid-size 10 --episodes 5000
```

### 啟動視覺化介面

```bash
python app.py
```

---

## 📁 專案結構 (規劃)

```
DRL_FinalProject/
├── README.md                # 專案說明文件
├── requirements.txt         # Python 依賴套件
├── train.py                 # 模型訓練腳本
├── evaluate.py              # 模型評估與基準測試
├── app.py                   # 視覺化介面主程式
├── envs/
│   └── grid_world.py        # 自定義 Gymnasium 環境
├── models/
│   └── d3qn.py              # D3QN 模型定義
├── utils/
│   ├── visualization.py     # 視覺化工具函數
│   └── shap_explainer.py    # SHAP 解釋器
├── experiments/
│   ├── baseline_astar.py    # A* 基準測試
│   └── ablation_study.py    # 消融實驗
└── results/
    ├── models/              # 訓練好的模型權重
    └── figures/             # 實驗結果圖表
```

---

## 📄 授權

本專案僅供學術研究使用。

---

<p align="center">
  <i>Explainable AI × Reinforcement Learning × Autonomous Navigation</i>
</p>
