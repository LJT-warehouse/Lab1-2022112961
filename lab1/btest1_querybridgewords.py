"""
该模块包含测试用例类TestQueryBridgeWords，
用于测试DirectedGraph类在特定文本预处理下的表现。
"""
import unittest
from lab_01 import DirectedGraph, preprocess_text


class TestQueryBridgeWords(unittest.TestCase):
    """
    测试查询桥接词功能的测试用例类。
    该类包含了一系列测试方法，用于验证DirectedGraph类在特定文本预处理下的表现。
    """
    def setUp(self):
        # 测试用例文本
        text = (
            "The scientist carefully analyzed the data, wrote a detailed "
            "report, and shared the report with the team, but the team "
            "requested more data, so the scientist analyzed it again."
        )
        words = preprocess_text(text)
        self.graph = DirectedGraph()
        for i in range(len(words) - 1):
            self.graph.add_edge(words[i], words[i + 1])

    def test_bridge_words_the_scientist(self):
        """
        测试查询"The"和"scientist"之间的桥接词。
        由于文本中"The scientist"是连续出现的，预期结果是没有桥接词。
        """
        # 黑盒测试：查询"The"和"scientist"的桥接词
        result = self.graph.queryBridgeWords("The", "scientist")
        # 由于文本中有 "the scientist"，所以没有桥接词
        self.assertIn("没有桥接词", result)


if __name__ == '__main__':
    unittest.main()
