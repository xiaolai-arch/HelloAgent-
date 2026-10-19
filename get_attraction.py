import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市天气，使用Tavily Search API搜索并返回优化后的景点推荐
    :param city:
    :param weather:
    :return:
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get('TAVILY_API_KEY')
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查找
    query = f"'{city}'在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        # 4. 调用API
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        print(1)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"