import requests

def get_weather(city: str) -> str:
    """
    通过调用wttr.in API查询真实天气情况
    """
    #API 端点，我们请求json格式数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态是否为200
        response.raise_for_status()
        # 解析返回的json数据
        data = response.json()

        # 提取当天天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}, 气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return (f"错误：查询天气石 - {e}")
    except (KeyError, IndexError) as e:
        # 处理数据失败，可能是城市名称错误
        return f"错误：解析天气数据失败，可能是城市名称失效 - {e}"