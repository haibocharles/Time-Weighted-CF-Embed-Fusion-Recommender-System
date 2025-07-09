import pickle
import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any


class HybridRecommender:
    def __init__(
            self,
            i2i_sim: Dict[int, Dict[int, float]],  # 基於協同過濾的項目-項目相似度字典
            i2i_embsim: Dict[int, Dict[int, float]],  # 基於嵌入的項目-項目相似度字典
            item_interaction_count: Dict[int, int],  # 項目互動次數統計
            item_mean_timestamp: Dict[int, float],  # 項目平均互動時間戳
            item_topk_click: List[int],  # 熱門項目列表（按點擊量排序）
            cf_weight: float = 0.7,  # 協同過濾默認權重
            emb_weight: float = 0.3,  # 嵌入相似度默認權重
            sim_item_topk: int = 20,  # 每個項目考慮的相似項目數量
            recall_item_num: int = 10,  # 最終召回的推薦項目數量
            alpha: float = 0.5  # 時間衰減係數
    ):
        # 初始化模型參數
        self.i2i_sim = i2i_sim
        self.i2i_embsim = i2i_embsim
        self.item_interaction_count = item_interaction_count
        self.item_mean_timestamp = item_mean_timestamp
        self.item_topk_click = item_topk_click
        self.cf_weight = cf_weight
        self.emb_weight = emb_weight
        self.sim_item_topk = sim_item_topk
        self.recall_item_num = recall_item_num
        self.alpha = alpha
        # 計算數據集的最大時間戳（用於時間衰減計算）
        self.dataset_max_time = max(item_mean_timestamp.values()) if item_mean_timestamp else 0

        print(f"Dataset max timestamp initialized: {self.dataset_max_time}")
        # 計算兩種相似度矩陣的平均值
        self.calculate_sim_means()

    def calculate_sim_means(self):
        """計算兩種相似度矩陣的全局平均值"""
        # 計算協同過濾相似度平均值
        self.i2i_sim_mean = 0
        count = 0
        for sim_dict in self.i2i_sim.values():
            if sim_dict:
                self.i2i_sim_mean += np.mean(list(sim_dict.values()))
                count += 1
        self.i2i_sim_mean = self.i2i_sim_mean / count if count > 0 else 0

        # 計算嵌入相似度平均值
        self.i2i_embsim_mean = 0
        count = 0
        for sim_dict in self.i2i_embsim.values():
            if sim_dict:
                self.i2i_embsim_mean += np.mean(list(sim_dict.values()))
                count += 1
        self.i2i_embsim_mean = self.i2i_embsim_mean / count if count > 0 else 0

    def get_dynamic_weights(self, item_id: int) -> Tuple[float, float]:
        """根據項目的互動次數動態調整權重（熱門項目更依賴協同過濾）"""
        interaction_count = self.item_interaction_count.get(item_id, 0)
        # 互動次數越多，CF權重越高（上限0.95）
        cf_w = min(0.95, self.cf_weight + 0.1 * math.log(1 + interaction_count))
        emb_w = 1 - cf_w
        return cf_w, emb_w

    def get_time_decay_factor(self, item_id: int, current_time: float) -> float:
        """计算时间衰減因子（新项目获得更高权重）"""
        # 使用传入的当前时间作为基准
        current_timestamp= current_time

        # 获取项目的平均时间戳
        mean_time = self.item_mean_timestamp.get(item_id, current_timestamp)

        # 计算时间差（转换为天）
        time_diff_days = (current_timestamp - mean_time) / (24 * 3600 * 1000)

        # 应用指数衰減公式
        return math.exp(-self.alpha * time_diff_days)

    def hybrid_recommend(
            self,
            user_id: int,
            user_hist_items: List[Tuple[int, float]],  # 用戶歷史互動記錄(項目ID, 時間戳)
            current_timestamp: float  # 當前時間戳
    ) -> List[Tuple[int, float]]:  # 返回推薦列表(項目ID, 分數)

        # 處理無歷史記錄的情況（返回熱門項目）
        if not user_hist_items:
            return [(item, -i) for i, item in enumerate(self.item_topk_click[:self.recall_item_num])]

        # 提取用戶歷史互動的項目集合和時間戳
        user_hist_set = {item_id for item_id, _ in user_hist_items}
        hist_timestamps = [t for _, t in user_hist_items]

        # 計算用戶互動的時間範圍
        max_hist_time = max(hist_timestamps) if hist_timestamps else current_timestamp
        min_hist_time = min(hist_timestamps) if hist_timestamps else current_timestamp
        time_range = max(1, max_hist_time - min_hist_time)  # 防止除零錯誤

        # 初始化項目分數字典
        item_scores = defaultdict(float)

        # 遍歷用戶歷史互動記錄（位置索引和具體記錄）
        for loc, (item_id, click_time) in enumerate(user_hist_items):
            # 計算時間權重（最近發生的互動更重要）
            time_weight = 1.0 - (max_hist_time - click_time) / time_range

            # 合併相似項目的容器
            similar_items = {}

            # 1. 獲取協同過濾的相似項目
            cf_sim_items = self.i2i_sim.get(item_id, {})
            for j, sim_score in cf_sim_items.items():
                if j not in user_hist_set:  # 排除用戶已互動的項目
                    similar_items[j] = similar_items.get(j, 0.0) + sim_score * self.cf_weight

            # 2. 獲取嵌入相似的項目
            emb_sim_items = self.i2i_embsim.get(item_id, {})
            for j, sim_score in emb_sim_items.items():
                if j not in user_hist_set:  # 排除用戶已互動的項目
                    similar_items[j] = similar_items.get(j, 0.0) + sim_score * self.emb_weight

            # 取TopK相似項目（按組合分數排序）
            top_similar = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)[:self.sim_item_topk]

            # 為每個相似項目計算最終分數
            for j, combined_score in top_similar:
                # 獲取動態權重
                cf_w, emb_w = self.get_dynamic_weights(j)
                # 獲取時間衰減因子
                time_decay = self.get_time_decay_factor(j, current_timestamp)
                # 累計項目分數 = 組合相似度 × 時間權重 × 時間衰減 × 動態權重
                item_scores[j] += combined_score * time_weight * time_decay * (cf_w + emb_w)

        # 補充召回：如果推薦不足，添加熱門項目
        if len(item_scores) < self.recall_item_num:
            for i, popular_item in enumerate(self.item_topk_click):
                if popular_item not in item_scores and popular_item not in user_hist_set:
                    # 設置低於個性化推薦的分數（-i -100）
                    item_scores[popular_item] = -i - 100
                    if len(item_scores) >= self.recall_item_num:
                        break

        # 返回TopN推薦（按分數降序）
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:self.recall_item_num]


# ================ 輔助函數 ================
def load_recommender_model(model_path: str) -> HybridRecommender:
    """加載已保存的推薦模型"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def save_recommender_model(model: HybridRecommender, model_path: str):
    """保存推薦模型到文件"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_data_artifacts(data_dir: str) -> Dict[str, Any]:
    """加載所有必要的數據文件"""
    artifacts = {}
    # 路徑組合
    print(f"正在从以下目录加载数据文件: {data_dir}")
    cf_path = os.path.join(data_dir, 'itemcf_i2i_sim.pkl')
    emb_path = os.path.join(data_dir, 'itemembedding_i2i_sim.pkl')
    stats_path = os.path.join(data_dir, 'item_stats.pkl')

    # 1. 加載ItemCF相似度
    artifacts['i2i_sim'] = {}
    if os.path.exists(cf_path):
        with open(cf_path, 'rb') as f:
            artifacts['i2i_sim'] = pickle.load(f)

    # 2. 加載Embedding相似度
    artifacts['i2i_embsim'] = {}
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            artifacts['i2i_embsim'] = pickle.load(f)

    # 3. 加載項目統計信息
    artifacts['item_interaction_count'] = {}
    artifacts['item_mean_timestamp'] = {}
    artifacts['item_topk_click'] = []
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats_data = pickle.load(f)
            artifacts['item_interaction_count'] = stats_data.get('interaction_count', {})
            artifacts['item_mean_timestamp'] = stats_data.get('mean_timestamp', {})
            artifacts['item_topk_click'] = stats_data.get('topk_click', [])

    return artifacts


def calculate_item_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """從DataFrame計算項目統計數據"""
    return {
        'interaction_count': df['click_article_id'].value_counts().to_dict(),
        'mean_timestamp': df.groupby('click_article_id')['click_timestamp'].mean().to_dict(),
        'topk_click': df['click_article_id'].value_counts().index.tolist()[:100]  # 取Top100熱門項目
    }


def save_item_stats(stats: Dict[str, Any], data_dir: str):
    """保存項目統計數據到文件"""
    with open(os.path.join(data_dir, 'item_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)


def create_recommender_from_artifacts(artifacts: Dict[str, Any], **kwargs) -> HybridRecommender:
    """使用加載的數據工件創建推薦器實例"""
    return HybridRecommender(
        i2i_sim=artifacts['i2i_sim'],
        i2i_embsim=artifacts['i2i_embsim'],
        item_interaction_count=artifacts['item_interaction_count'],
        item_mean_timestamp=artifacts['item_mean_timestamp'],
        item_topk_click=artifacts['item_topk_click'],
        **kwargs
    )


def initialize_recommender(train_data_path: str, data_dir: str,model_dir: str = None, save_model: bool = True) -> HybridRecommender:
    # 1. 加載訓練數據
    print(f"Loading training data from {train_data_path}...")
    df = pd.read_csv(train_data_path)

    # 2. 計算並保存項目統計
    print("Calculating item statistics...")
    stats = calculate_item_stats(df)
    save_item_stats(stats, data_dir)

    # 3. 加載所有數據工件
    print("Loading data artifacts...")
    artifacts = load_data_artifacts(data_dir)

    # 4. 創建推薦器實例
    print("Creating recommender instance...")
    recommender = create_recommender_from_artifacts(artifacts)

    # 5. 保存模型（如果启用）
    if save_model:
        # 如果没有指定model_dir，则使用data_dir的父目录下的models文件夹
        if model_dir is None:
            base_dir = os.path.dirname(data_dir)  # 获取data_dir的父目录
            model_dir = os.path.join(base_dir, "models")

        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'hybrid_recommender.pkl')
        print(f"Saving model to {model_path}...")
        save_recommender_model(recommender, model_path)

    print("Recommender initialization completed!")
    return recommender
