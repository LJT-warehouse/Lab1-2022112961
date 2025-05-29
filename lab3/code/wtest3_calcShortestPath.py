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

    def test_calc_shortest_path_start_word_not_in_graph(self):
        """测试用例3：起始单词不在图中"""
        result = self.graph.calcShortestPath("unknown")
        expected = "错误：起始单词 'unknown' 不在图中"
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
