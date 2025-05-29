import unittest
from lab_01 import DirectedGraph, preprocess_text


class TestCalcShortestPath(unittest.TestCase):
    def setUp(self):
        # 初始化测试用的有向图
        self.graph = DirectedGraph()
        text = (
            "The scientist carefully analyzed the data, wrote a detailed "
            "report, and shared the report with the team, but the team "
            "requested more data, so the scientist analyzed it again."
        )
        words = preprocess_text(text)
        for i in range(len(words) - 1):
            self.graph.add_edge(words[i], words[i + 1])

    def test_calc_shortest_path_two_words(self):
        """测试用例2：输入两个单词，计算它们之间的最短路径"""
        result = self.graph.calcShortestPath("The", "scientist")
        expected = "从 'the' 到 'scientist' 的最短路径是:\nthe -> scientist\n总权重: 2.00"
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
