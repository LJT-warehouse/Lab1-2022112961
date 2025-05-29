"""
这是一个测试模块，用于测试lab_01中的DirectedGraph类和preprocess_text函数。
该模块包含了多个测试用例类，每个类中定义了不同的测试场景。
"""
import unittest
from lab_01 import DirectedGraph, preprocess_text


class TestQueryBridgeWordsEmptyAgain(unittest.TestCase):
    """
    测试在查询桥接词时，再次处理文本的场景。
    设置了一个包含科学家分析数据、编写报告并再次分析的文本，测试DirectedGraph类的功能。
    """
    def setUp(self):
        text = (
            "The scientist carefully analyzed the data, wrote a detailed "
            "report, and shared the report with the team, but the team "
            "requested more data, so the scientist analyzed it again."
        )
        words = preprocess_text(text)
        self.graph = DirectedGraph()
        for i in range(len(words) - 1):
            self.graph.add_edge(words[i], words[i + 1])

    def test_empty_and_again(self):
        """
        测试查询桥接词时，起始单词为空的情况。
        结果应包含提示信息“单词不能为空”。
        """
        result = self.graph.queryBridgeWords("", "again")
        self.assertIn("单词不能为空", result)


if __name__ == '__main__':
    unittest.main()
