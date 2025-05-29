"""
这个模块包含了一个测试类TestQueryBridgeWordsUnknownScientist，
用于测试lab_01模块中的DirectedGraph类和preprocess_text函数。
"""
import unittest
from lab_01 import DirectedGraph, preprocess_text


class TestQueryBridgeWordsUnknownScientist(unittest.TestCase):
    """测试 DirectedGraph 类和 preprocess_text 函数在处理未知科学家数据时的行为。"""
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

    def test_unknown_and_scientist(self):
        """测试未知科学家的查询。"""
        result = self.graph.queryBridgeWords("unknown", "scientist")
        result = self.graph.queryBridgeWords("unknown", "scientist")
        self.assertIn("不在图中", result)


if __name__ == '__main__':
    unittest.main()
