"""
所有后端必须实现的统一接口。

这样 main.py 只需面向这个接口编程，不关心具体是哪个框架在跑。
每个后端自管对话状态（多轮上下文），main.py 只负责循环 + 打印。
"""

from abc import ABC, abstractmethod


class AgentBackend(ABC):
    @abstractmethod
    def reply(self, user_input: str) -> str:
        """处理一条用户输入，返回最终文本回复。

        后端内部负责：调模型 → 必要时调工具 → 回填结果 → 维护多轮上下文。
        需要打印中间过程（如工具调用）的后端，自行在 reply 内打印。
        """

    @property
    def name(self) -> str:
        """后端显示名，用于启动横幅（可选覆盖）。"""
        return type(self).__name__
