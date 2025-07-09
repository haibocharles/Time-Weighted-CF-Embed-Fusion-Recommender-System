from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from fastapi.staticfiles import StaticFiles
import os
import pickle
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator, ValidationError,Field
import time
from recommender import load_recommender_model, initialize_recommender
from fastapi.responses import HTMLResponse,FileResponse
# 定义支持的时间格式
SUPPORTED_TIME_FORMATS = [
    "%Y-%m-%d",                  # 年月日 (2025-01-01)
    "%Y-%m-%d %H:%M",            # 年月日小时分钟 (2025-01-01 12:00)
    "%Y-%m-%d %H:%M:%S",         # 年月日小时分钟秒 (2025-01-01 12:00:00)
    "%Y/%m/%d",                  # 年月日 (2025/01/01)
    "%Y/%m/%d %H:%M",            # 年月日小时分钟 (2025/01/01 12:00)
    "%Y/%m/%d %H:%M:%S",         # 年月日小时分钟秒 (2025/01/01 12:00:00)
    "%Y年%m月%d日",              # 年月日 (2025年01月01日)
    "%Y年%m月%d日 %H:%M",        # 年月日小时分钟 (2025年01月01日 12:00)
    "%Y年%m月%d日 %H:%M:%S",     # 年月日小时分钟秒 (2025年01月01日 12:00:00)
]
async def lifespan(app : FastAPI): #把FastAPI綁定給app 上
    global recommender
    """ deal with application startup and shutdown events """
    try:
        if os.path.exists(MODEL_PATH):
            print("Loading pre-trained model...")
            recommender = load_recommender_model(MODEL_PATH)
            print("Model loaded successfully")
        else:
            print("Initializing new model...")
            recommender = initialize_recommender(TRAIN_DATA_PATH, DATA_DIR)
            print("Model initialized successfully")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        recommender = None
        raise e
    yield  # 當 API 啟動時它會加載模型，當 API 停止時它會清理資源


# 初始化FastAPI应用
app = FastAPI( # API 物件
    title="recommender system API",
    description="Hybrid Recommendation System Based on Collaborative Filtering and Content Embedding",#基于协同过滤和内容嵌入的混合推荐系统
    version="1.0.0",
    lifespan=lifespan  # 保持这个参数
)

# 配置路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#根目錄
print(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data') # data資料夾
print(DATA_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'models')#models資料夾
print(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_recommender.pkl')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_click_log.csv')
Page_DIR = os.path.join(BASE_DIR, 'static') # 靜態文件夾
# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(Page_DIR, exist_ok=True) # 确保静态文件夹存在

#訪問頁面
#app.mount 表示将静态文件夹挂载到应用上
# app.mount("/static", StaticFiles(directory="static"), name="static")
# 当前活动模型
recommender = None


# 时间转换函数
def convert_to_timestamp_ms(time_str: Optional[str]) -> Optional[float]:
    """将各种格式的时间字符串转换为毫秒时间戳"""
    if time_str is None:
        return None

    # 尝试所有支持的时间格式
    for fmt in SUPPORTED_TIME_FORMATS:
        try:
            dt = datetime.strptime(time_str, fmt)# 将字符串解析为datetime对象
            # 如果时间没有时区信息，假设为UTC时间
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp() * 1000  # 转换为毫秒
        except ValueError:
            continue

    # 解析ISO格式
    try:
        dt = datetime.fromisoformat(time_str) # 将ISO格式字符串解析为datetime对象
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000
    except ValueError:
        pass

    # 尝试解析时间戳字符串
    try:
        # 可能是毫秒时间戳字符串
        return float(time_str)
    except ValueError:
        pass

    # 所有尝试都失败
    raise ValueError(f"Invalid time format please try again: {time_str}")

# 请求/响应模型
class RecommendRequest(BaseModel):
    user_id: int = Field(example=123456, description="用户唯一标识符")
    user_history: List[Dict[str, Any]] = Field(
        example=[{"article_id": 456, "timestamp": "2025-01-01 12:00"}],
    )
    current_timestamp: Optional[float] = Field(
        default=None,
    )

    @field_validator('current_timestamp', mode='before')
    def validate_current_timestamp(cls, v):
        """验证并转换current_timestamp字段"""
        if v is None:
            return None
        try:
            return convert_to_timestamp_ms(str(v))
        except ValueError as e:
            raise ValueError(f"Invalid time format: {v}. {str(e)}")

    @field_validator('user_history', mode='before')
    def validate_user_history(cls, v):
        """验证并转换user_history中的时间戳"""
        if not isinstance(v, list):
            raise ValueError("user_history must be a list")

        for item in v:
            if not isinstance(item, dict):
                raise ValueError("Each item in user_history must be a dictionary")
            if 'article_id' not in item or 'timestamp' not in item:
                raise ValueError("Each item must contain 'article_id' and 'timestamp'")

            try:
                item['timestamp'] = convert_to_timestamp_ms(item['timestamp'])
            except ValueError as e:
                raise ValueError(f"Invalid timestamp: {item['timestamp']}. {str(e)}")

        return v




# 響應模型
class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]] # [{"article_id": 456, "score": 0.85}]
    messages: List[str] = []  # 添加消息字段





@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):

    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model is not loaded")
    user_hist = []
    messages = []
    if not request.user_history:
        messages.append("Cold-start: Because the user has no history, the system will return popular items.")
    else:
        for item in request.user_history:
            if 'article_id' not in item or 'timestamp' not in item:
                messages.append("Invalid history item format, please provide valid 'article_id' or 'timestamp'.")
                continue
            try:
                timestamp = float(item['timestamp']) if isinstance(item['timestamp'], (int, float)) \
                    else convert_to_timestamp_ms(item['timestamp'])
                user_hist.append((item['article_id'], timestamp))
            except Exception as e:
                messages.append(f"Invalid timestamp: {str(e)}")


    # 处理当前时间戳
    current_timestamp = request.current_timestamp
    if current_timestamp is None:
        current_timestamp = recommender.dataset_max_time # 如果未提供，则使用数据集中的最大时间戳
        messages.append("No current timestamp provided, current timestamp replaced by the dataset max time.")

    elif current_timestamp < recommender.dataset_max_time:
        # 如果提供了当前时间戳且小于数据集最大时间，则使用提供的时间戳
        messages.append("System can use provided current timestamp.")
    else:
        # 如果提供了現在时间戳，但大于数据集最大时间，则使用数据集最大时间
        current_timestamp = min(current_timestamp, recommender.dataset_max_time)
        messages.append("Since the current timestamp exceeds the dataset's maximum time, the system will use the dataset's maximum time instead.")
    try:
        recommendations = recommender.hybrid_recommend(
            request.user_id,
            user_hist,
            current_timestamp
        )

        # 确保所有类型都是可序列化的
        formatted_recs = [{
            "article_id": int(item_id),
            "score": round(score, 10)
        } for item_id, score in recommendations]

        return {
            "user_id": int(request.user_id),
            "recommendations": formatted_recs,
            "messages": messages[:3]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")




# 重新加载模型端点
@app.get("/reload-model")
async def reload_model():
    global recommender
    try:
        recommender = initialize_recommender(TRAIN_DATA_PATH, DATA_DIR)
        return {"status": "success", "message": "successfully reloaded the recommendation model"}
    except Exception as e:
        return {"status": "error", "message": f"failed to reloaded the recommendation model: {str(e)}"}


# 健康检查端点
@app.get("/health")
async def health_check():
    return {
        "status": "ok" if recommender is not None else "unavailable",
        "model_loaded": recommender is not None
    }

# 静态文件路由
app.mount("/static", StaticFiles(directory="../static"), name="static")
# 首页
@app.get("/Main_page", response_class=HTMLResponse)
async def home():
    # 直接返回静态目录中的HTML文件
    return FileResponse("../static/Main_page.html")

@app.get("/")
async def home():
    return {
        "message": "Recommender System API is running and please visit http://0.0.0.0:5726/Main_page to use this Recommender System, Thanks!",
        "endpoints": {
            "/recommend": "POST get recommendations",
            "/reload-model": "GET reload the recommendation model",
            "/health": "GET service health check"
        }
    }