"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _create_with_param_fallback(self, kwargs: dict):
        """Call chat.completions.create, auto-dropping unsupported params.

        Reasoning models (o1, gpt-5, etc.) reject temperature and only
        accept max_completion_tokens (not max_tokens). They also consume
        large amounts of reasoning tokens, so when temperature is rejected
        we also remove max_completion_tokens to avoid empty responses due
        to reasoning token budget exhaustion.
        """
        _PARAM_FALLBACKS = {
            "max_completion_tokens": "max_tokens",  # try legacy name
            "max_tokens": None,                     # just drop
            "temperature": None,
        }
        is_reasoning_model = False
        while True:
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                err = str(e)
                if "Unsupported" not in err and "not supported" not in err:
                    raise
                # Find which param was rejected
                dropped = False
                for param, fallback in _PARAM_FALLBACKS.items():
                    if param in err and param in kwargs:
                        val = kwargs.pop(param)
                        if param == "temperature":
                            # Temperature rejected = reasoning model.
                            # Remove token limits too, since reasoning
                            # tokens consume most of the budget.
                            is_reasoning_model = True
                            kwargs.pop("max_completion_tokens", None)
                            kwargs.pop("max_tokens", None)
                        elif fallback and fallback not in kwargs:
                            kwargs[fallback] = val
                        dropped = True
                        break
                if not dropped:
                    raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Some models (o1, gpt-5, etc.) don't support certain params like
        # temperature, max_tokens, max_completion_tokens. Retry by stripping
        # the unsupported parameter on each 400 error.
        response = self._create_with_param_fallback(kwargs)

        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format returned by LLM: {cleaned_response}")

