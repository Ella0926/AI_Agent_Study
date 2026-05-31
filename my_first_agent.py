import os
import re
import requests
from openai import OpenAI
from tavily import TavilyClient

# ==========================================
# 1. 配置最高宪法：指令模板（System Prompt）
# ==========================================
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

# ==========================================
# 2. 编写实体工具函数（Tools）
# ==========================================

def get_weather(city: str) -> str:
    """通过调用 wttr.in API 查询真实的天气信息。"""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    except Exception as e:
        return f"错误:查询天气时遇到问题 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。"""
    # 老师的代码里是从环境变量读取 KEY
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]
        
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        return "\n".join(formatted_results) if formatted_results else "抱歉，没有找到相关的旅游景点推荐。"
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# 老师在 1.3.1 末尾提到的：将工具放入字典映射，供后续主循环自动化反射调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


# ==========================================
# 3. 接入大语言模型：老师的接口兼容类实现 (1.3.2)
# ==========================================
class OpenAICompatibleClient:
    """一个用于调用任何兼容OpenAI接口的LLM服务的客户端。"""
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"错误:调用语言模型服务时出错。{e}"


# ==========================================
# 4. 执行行动循环：核心主循环控制流 (1.3.3)
# ==========================================
if __name__ == "__main__":
    # 🛠️ 核心配置区：请在这里填入你的真实凭证 🛠️
    # 如果你使用 DeepSeek，可以把 BASE_URL 换成 https://api.deepseek.com
    # 🛠️ 核心配置区：请在这里填入你的真实凭证 🛠️
    # 如果你使用 DeepSeek，可以把 BASE_URL 换成 https://api.deepseek.com
    # 🛠️ 核心配置区：换成智谱 AI 的免费全套参数 🛠️
    API_KEY = "164eb5efac704ee59f48c31154431ffc.IX6TOgKcxsgdaoTg"
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # 👈 必须换成智谱大楼的官方正门地址
    MODEL_ID = "glm-4-flash"                          # 👈 换成老师推荐的、又快又聪明的轻量大脑型号              # 👈 将模型 ID 换成 deepseek-chat

    # 将 Tavily 的 KEY 直接注入环境变量
    os.environ['TAVILY_API_KEY'] = "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" # 👈 填入你从 Tavily 官网后台复制出来的 tvly- 开头的完整密钥

    # 实例化大模型大脑
    llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

    # 初始用户输入
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "="*40)

    # 运行主循环流程（限制最大步数防止无限套娃死循环）
    for i in range(5):
        print(f"--- 循环 {i+1} ---\n")
        full_prompt = "\n".join(prompt_history)
        
        # 1. 思考 (Thought)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        
        # 截断可能多出的冗余文本，强行规范
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            llm_output = match.group(1).strip()
            
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)
        
        # 2. 解析并执行行动 (Action)
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation_str = "Observation: 错误: 未能解析到 Action 字段。请严格遵循规定格式。"
            prompt_history.append(observation_str)
            continue
            
        action_str = action_match.group(1).strip()

        # 如果大模型觉得大功告成了，发出 Finish 信号，跳出循环
        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"任务完成！\n最终答案: {final_answer}")
            break
        
        # 利用正则表达式，把 Action: get_weather(city="北京") 拆解成函数名和参数
        try:
            tool_name = re.search(r"(\w+)\(", action_str).group(1)
            args_str = re.search(r"\((.*)\)", action_str).group(1)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

            # 3. 感知反馈 (Observation)
            if tool_name in available_tools:
                # 动态反射执行对应的 Python 函数
                observation = available_tools[tool_name](**kwargs)
            else:
                observation = f"错误:未定义的工具 '{tool_name}'"
        except Exception as e:
            observation = f"错误:解析行动指令失败 - {e}"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        # 把这次从物理世界拿到的“新鲜食材”，喂进历史记录，供下一轮循环 LLM 来看
        prompt_history.append(observation_str)