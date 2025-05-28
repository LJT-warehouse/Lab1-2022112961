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

    def test_calc_shortest_path_single_word(self):
        """测试用例1：只输入一个单词，计算到所有其他节点的最短路径"""
        result = self.graph.calcShortestPath("The")
        self.assertIn("从 'the' 到", result)
        self.assertIn("的最短路径是", result)


if __name__ == '__main__':
    unittest.main()
