import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Lab1 实验：基于大模型的编程与Git实战

import re
import random
import sys
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button
import matplotlib.patches as patches
import numpy as np

class DirectedGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(int))
        self.fig = None
        self.ax = None
        self.pos = None
        self.G = None
        self.dragging_node = None
        self.drag_start_pos = None
        self.node_size = 2000  # 统一节点大小

    def add_edge(self, from_node, to_node):
        self.graph[from_node][to_node] += 1

    def get_nodes(self):
        return list(self.graph.keys())

    def get_edges(self):
        edges = []
        for u in self.graph:
            for v in self.graph[u]:
                edges.append((u, v, self.graph[u][v]))
        return edges

    def showDirectedGraph(self):
        self.G = nx.DiGraph()
        for u in self.graph:
            for v, w in self.graph[u].items():
                self.G.add_edge(u, v, weight=w)
        
        # CLI输出
        self.showDirectedGraphCLI()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.pos = nx.spring_layout(self.G, k=1, iterations=50)
        self.ax.clear()

        self.node_collection = nx.draw_networkx_nodes(
            self.G, self.pos, 
            node_color='lightblue',
            node_size=self.node_size,
            alpha=0.9,
            ax=self.ax
        )
        self._draw_edges_with_arrows()
        nx.draw_networkx_labels(self.G, self.pos, font_size=10, font_weight='bold', ax=self.ax)
        self._draw_edge_labels()
        plt.title("Directed Graph Visualization", fontsize=16, pad=20)
        plt.axis('off')
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        save_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
        save_button = Button(save_ax, '保存图形')
        save_button.on_clicked(self.save_graph)
        plt.show()

    def showDirectedGraphCLI(self):
        """在命令行界面显示有向图的结构"""
        print("\n=== 有向图结构 ===")
        
        # 获取所有节点
        nodes = sorted(self.graph.keys())
        
        # 为每个节点创建其出边的表示
        for node in nodes:
            if node in self.graph and self.graph[node]:
                # 获取当前节点的所有出边
                edges = []
                for target, weight in self.graph[node].items():
                    edge = f"{node}→{target}"
                    if weight > 1:
                        edge += f"({weight})"
                    edges.append(edge)
                
                # 打印当前节点的所有出边
                print("  ".join(edges))
        
        print("=" * 20)
        print("说明：")
        print("- 箭头(→)表示边的方向")
        print("- 括号中的数字表示边的权重")
        print("=" * 20)

    def _draw_edges_with_arrows(self):
        # 计算节点半径（以数据坐标为单位）
        node_radius = np.sqrt(self.node_size/np.pi) / 1000  # 经验缩放，适配layout
        for u, v, data in self.G.edges(data=True):
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy)
            if dist == 0:
                continue
            # 两端都缩进节点半径
            shrink_ratio = node_radius / dist
            start_x = x1 + dx * shrink_ratio
            start_y = y1 + dy * shrink_ratio
            end_x = x2 - dx * shrink_ratio
            end_y = y2 - dy * shrink_ratio
            # 再次缩进终点，保证箭头尾部也不在圆内
            start_x = start_x + dx * shrink_ratio
            start_y = start_y + dy * shrink_ratio
            end_x = end_x - dx * shrink_ratio
            end_y = end_y - dy * shrink_ratio
            # 绘制箭头
            arrow = patches.FancyArrowPatch(
                (start_x, start_y), (end_x, end_y),
                arrowstyle='-|>',
                color='gray',
                mutation_scale=20,
                linewidth=1.5,
                connectionstyle='arc3,rad=0.1',
                zorder=1
            )
            self.ax.add_patch(arrow)

    def _draw_edge_labels(self):
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        for (u, v), label in edge_labels.items():
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2
            self.ax.text(label_x, label_y, str(label), color='black', fontsize=9, ha='center', va='center', backgroundcolor='white')

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        for node, (x, y) in self.pos.items():
            if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                self.dragging_node = node
                self.drag_start_pos = (event.xdata, event.ydata)
                break

    def on_release(self, event):
        self.dragging_node = None
        self.drag_start_pos = None

    def on_motion(self, event):
        if self.dragging_node is None or event.inaxes != self.ax:
            return
        self.pos[self.dragging_node] = (event.xdata, event.ydata)
        self.ax.clear()
        self.node_collection = nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color='lightblue',
            node_size=self.node_size,
            alpha=0.9,
            ax=self.ax
        )
        self._draw_edges_with_arrows()
        nx.draw_networkx_labels(self.G, self.pos, font_size=10, font_weight='bold', ax=self.ax)
        self._draw_edge_labels()
        plt.title("Directed Graph Visualization", fontsize=16, pad=20)
        plt.axis('off')
        self.fig.canvas.draw()

    def save_graph(self, event):
        try:
            plt.savefig("graph.png", format="png", dpi=300, bbox_inches='tight')
            print("图形已保存为 graph.png")
        except Exception as e:
            print(f"保存图片时出错: {e}")

    def queryBridgeWords(self, word1, word2):
        word1, word2 = word1.lower(), word2.lower()
        if word1 not in self.graph:
            return f"错误：单词 '{word1}' 不在图中"
        if word2 not in self.graph:
            return f"错误：单词 '{word2}' 不在图中"
        
        bridges = [w for w in self.graph[word1] if word2 in self.graph[w]]
        if not bridges:
            return f"在 '{word1}' 和 '{word2}' 之间没有桥接词"
        elif len(bridges) == 1:
            return f"从 '{word1}' 到 '{word2}' 的桥接词是: {bridges[0]}"
        else:
            return f"从 '{word1}' 到 '{word2}' 的桥接词有: {', '.join(bridges)}"

    def generateNewText(self, inputText):
        words = preprocess_text(inputText)
        if len(words) < 2:
            return "错误：输入文本太短，至少需要两个单词"
        
        new_text = []
        for i in range(len(words) - 1):
            new_text.append(words[i])
            bridges = [w for w in self.graph[words[i]] if words[i + 1] in self.graph[w]]
            if bridges:
                new_text.append(random.choice(bridges))
        new_text.append(words[-1])
        return ' '.join(new_text)

    def calcShortestPath(self, word1, word2=None):
        word1 = word1.lower()
        if word1 not in self.graph:
            return f"错误：起始单词 '{word1}' 不在图中"
        
        # 如果只输入了一个单词，计算到所有其他节点的最短路径
        if word2 is None:
            results = []
            for target in self.graph.keys():
                if target != word1:
                    result = self._calculate_single_path(word1, target)
                    if result:
                        results.append(result)
            if not results:
                return f"从 '{word1}' 到其他节点不存在任何路径"
            return "\n\n".join(results)
        
        # 如果输入了两个单词，计算它们之间的最短路径
        word2 = word2.lower()
        if word2 not in self.graph:
            return f"错误：目标单词 '{word2}' 不在图中"
        
        return self._calculate_single_path(word1, word2)

    def _calculate_single_path(self, word1, word2):
        """计算两个单词之间的最短路径的辅助方法"""
        dist = defaultdict(lambda: float('inf'))
        prev = {}
        dist[word1] = 0
        q = deque([word1])
        
        while q:
            u = q.popleft()
            for v in self.graph[u]:
                if dist[u] + self.graph[u][v] < dist[v]:
                    dist[v] = dist[u] + self.graph[u][v]
                    prev[v] = u
                    q.append(v)
        
        if word2 not in dist or dist[word2] == float('inf'):
            return None
        
        path = []
        cur = word2
        while cur != word1:
            path.append(cur)
            cur = prev[cur]
        path.append(word1)
        path.reverse()
        
        return f"从 '{word1}' 到 '{word2}' 的最短路径是:\n{' -> '.join(path)}\n总权重: {dist[word2]:.2f}"

    def calPageRank(self, d=0.85, max_iter=100, tol=1e-6):
        nodes = set(self.graph.keys()) | {v for u in self.graph for v in self.graph[u]}
        N = len(nodes)
        pr = dict.fromkeys(nodes, 1.0 / N)
        for _ in range(max_iter):
            new_pr = dict.fromkeys(nodes, (1 - d) / N)
            # 计算所有出度为0节点的PR总和
            dangling_sum = sum(pr[u] for u in nodes if len(self.graph[u]) == 0)
            for u in self.graph:
                out_sum = sum(self.graph[u].values())
                if out_sum > 0:
                    for v in self.graph[u]:
                        new_pr[v] += d * pr[u] * self.graph[u][v] / out_sum
            # 悬挂节点贡献均分给所有节点
            for v in nodes:
                new_pr[v] += d * dangling_sum / N
            if max(abs(new_pr[n] - pr[n]) for n in nodes) < tol:
                break
            pr = new_pr
        return pr

    def randomWalk(self):
        if not self.graph:
            return "错误：图为空"
        
        visited = set()
        current = random.choice(list(self.graph.keys()))
        walk = [current]
        
        while True:
            if not self.graph[current]:
                break
            next_node = random.choice(list(self.graph[current].keys()))
            edge = (current, next_node)
            if edge in visited:
                break
            visited.add(edge)
            walk.append(next_node)
            current = next_node
        
        result = ' '.join(walk)
        try:
            with open("random_walk.txt", "w", encoding='utf-8') as f:
                f.write(result)
        except Exception as e:
            print(f"警告：保存结果到文件时出错: {e}")
        
        return result


def preprocess_text(text):
    if not isinstance(text, str):
        raise ValueError("输入必须是字符串类型")
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    words = text.lower().split()
    return [w for w in words if w]  # 移除空字符串

def load_graph_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            raise ValueError("文件为空")
            
        graph = DirectedGraph()
        words = preprocess_text(text)
        if len(words) < 2:
            raise ValueError("文本太短，无法构建图")
            
        for i in range(len(words) - 1):
            graph.add_edge(words[i], words[i + 1])
        
        return graph
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {filepath}")
    except Exception as e:
        raise Exception(f"处理文件时出错: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("用法: python lab1.py <文件名>")
        print("示例: python lab1.py input.txt")
        return
    
    filepath = sys.argv[1]
    try:
        graph = load_graph_from_file(filepath)
        print(f"成功从文件 {filepath} 加载图形数据")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    while True:
        print("\n=== 文本图形分析系统 ===")
        print("1. 显示有向图")
        print("2. 查询桥接词")
        print("3. 生成新文本")
        print("4. 计算最短路径")
        print("5. 计算节点重要度(PageRank)")
        print("6. 随机游走")
        print("0. 退出程序")
        print("=" * 25)
        
        try:
            choice = input("请输入选项编号: ")
            if choice == '1':
                print("正在生成并显示有向图...")
                graph.showDirectedGraph()
            
            elif choice == '2':
                w1 = input("请输入第一个单词: ").strip()
                w2 = input("请输入第二个单词: ").strip()
                if not w1 or not w2:
                    print("错误：单词不能为空")
                    continue
                print(graph.queryBridgeWords(w1, w2))
            
            elif choice == '3':
                text = input("请输入要处理的文本: ").strip()
                if not text:
                    print("错误：输入文本不能为空")
                    continue
                print("生成的新文本:")
                print(graph.generateNewText(text))
            
            elif choice == '4':
                w1 = input("请输入起始单词: ").strip()
                if not w1:
                    print("错误：起始单词不能为空")
                    continue
                w2 = input("请输入目标单词(可选，直接回车则计算到所有其他节点的最短路径): ").strip()
                if w2:
                    print(graph.calcShortestPath(w1, w2))
                else:
                    print(graph.calcShortestPath(w1))
            
            elif choice == '5':
                print("计算PageRank值...")
                pr = graph.calPageRank()
                print("\nPageRank结果 (按重要度降序排列):")
                for k, v in sorted(pr.items(), key=lambda x: -x[1])[:10]:
                    print(f"{k}: {v:.4f}")
            
            elif choice == '6':
                print("随机游走结果:")
                result = graph.randomWalk()
                print(result)
                print("\n结果已保存到 random_walk.txt")
            
            elif choice == '0':
                print("感谢使用！再见！")
                break
            
            else:
                print("无效的选项，请重新输入")
        
        except Exception as e:
            print(f"操作出错: {e}")
            continue

if __name__ == "__main__":
    main()
