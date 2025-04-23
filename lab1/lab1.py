import re
import random
from collections import defaultdict
import heapq
import time
import math
import networkx as nx
import matplotlib.pyplot as plt


class DirectedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.weights = defaultdict(dict)

    def add_edge(self, u, v, weight):
        self.graph[u].append(v)
        self.weights[u][v] = weight

    def get_neighbors(self, node):
        return self.graph[node]

    def get_weight(self, u, v):
        return self.weights[u][v]

    def __contains__(self, node):
        return node in self.graph


def preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text


def build_graph_from_text(text):
    graph = DirectedGraph()
    words = text.split()
    word_count = defaultdict(int)

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        word_count[word1] += 1
        word_count[word2] += 1
        if word2 not in graph.get_neighbors(word1):
            graph.add_edge(word1, word2, 1)
        else:
            graph.add_edge(word1, word2, graph.get_weight(word1, word2) + 1)

    return graph, word_count


def show_directed_graph(graph):
    for node in graph.graph:
        neighbors = graph.get_neighbors(node)
        if neighbors:  # 只有当节点有出边时才输出
            print(f"{node} -> {', '.join(neighbors)}")


def query_bridge_words(graph, word1, word2):
    if word1 not in graph.graph:
        return f"No '{word1}' in the graph!"
    if word2 not in graph.graph:
        return f"No '{word2}' in the graph!"

    bridge_words = []
    for node in graph.graph:
        if node in graph.get_neighbors(word1) and word2 in graph.get_neighbors(node):
            bridge_words.append(node)

    if not bridge_words:
        return f"No bridge words from '{word1}' to '{word2}'!"
    return f"The bridge words from '{word1}' to '{word2}' are: {', '.join(bridge_words)}"


def generate_new_text(graph, input_text):
    words = input_text.split()
    new_text = []

    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        bridge_result = query_bridge_words(graph, word1, word2)

        if "No bridge words" in bridge_result:
            new_text.append(word1)
        elif "No" in bridge_result:
            new_text.append(word1)
        else:
            bridge_word = bridge_result.split("are: ")[1].split(", ")[0]
            new_text.append(word1)
            new_text.append(bridge_word)

    new_text.append(words[-1])
    return ' '.join(new_text)


def calc_shortest_path(graph, word1, word2=None):
    if word1 not in graph.graph:
        return f"No '{word1}' in the graph."

    distances = {node: float('inf') for node in graph.graph}
    previous_nodes = {node: None for node in graph.graph}

    distances[word1] = 0
    priority_queue = [(0, word1)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node == word2:
            break
        for neighbor in graph.get_neighbors(current_node):
            if neighbor not in distances:
                distances[neighbor] = float('inf')
            weight = graph.get_weight(current_node, neighbor)
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    if word2 is not None:
        if word2 not in distances or distances[word2] == float('inf'):
            return f"No path from '{word1}' to '{word2}'."
        path = []
        current_node = word2
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()
        return ' -> '.join(path), distances[word2]
    else:
        results = []
        for node in graph.graph:
            if node != word1 and distances[node] != float('inf'):
                path = []
                current_node = node
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                path.reverse()
                results.append(f"Shortest path from '{word1}' to '{node}': {' -> '.join(path)}, Length: {distances[node]}")
        return "\n".join(results)


def calculate_tf_idf(text):
    words = text.split()
    word_count = defaultdict(int)
    document_count = defaultdict(int)

    for word in words:
        word_count[word] += 1

    unique_words = set(words)
    for word in unique_words:
        document_count[word] += 1

    tf_idf = defaultdict(float)
    total_words = len(words)

    for word in word_count:
        tf = word_count[word] / total_words
        idf = math.log(len(unique_words) / (document_count[word] + 1))
        tf_idf[word] = tf * idf

    total_tfidf = sum(tf_idf.values())
    for word in tf_idf:
        tf_idf[word] /= total_tfidf

    return tf_idf


def cal_pagerank(graph, damping_factor=0.85, iterations=100):
    nodes = list(graph.graph.keys())
    num_nodes = len(nodes)

    # 计算 TF-IDF 作为初始 PR 值
    tf_idf = calculate_tf_idf(" ".join(nodes))

    # 初始化 PR 值
    rank = {node: tf_idf.get(node, 1.0 / num_nodes) for node in nodes}

    for _ in range(iterations):
        new_rank = {node: (1 - damping_factor) / num_nodes for node in nodes}
        for node in nodes:
            for neighbor in graph.get_neighbors(node):
                if neighbor not in new_rank:  # 确保邻居节点在 new_rank 中
                    new_rank[neighbor] = (1 - damping_factor) / num_nodes
                new_rank[neighbor] += damping_factor * (rank[node] / len(graph.get_neighbors(node)))
        rank = new_rank

    return rank


def random_walk(graph, output_file="random_walk.txt"):
    nodes = list(graph.graph.keys())
    if not nodes:
        return "The graph is empty."

    current_node = random.choice(nodes)
    walk_path = [current_node]

    visited_edges = defaultdict(set)

    try:
        while True:
            neighbors = graph.get_neighbors(current_node)
            if not neighbors:
                break

            unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited_edges[current_node]]

            if not unvisited_neighbors:
                if len(walk_path) > 1:
                    current_node = walk_path[-2]
                    walk_path.pop()
                    continue
                else:
                    break

            next_node = random.choice(unvisited_neighbors)
            walk_path.append(next_node)
            visited_edges[current_node].add(next_node)

            current_node = next_node

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nRandom walk stopped by user.")

    # 将路径保存到文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(walk_path))

    print(f"Random walk path saved to {output_file}")
    return ' '.join(walk_path)


def save_directed_graph_as_image(graph, output_file="directed_graph.png"):
    G = nx.DiGraph()
    for node in graph.graph:
        for neighbor in graph.get_neighbors(node):
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G)  # 使用弹簧布局
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
    plt.title("Directed Graph")
    plt.savefig(output_file)
    plt.show()
    print(f"Directed graph saved to {output_file}")


def main():
    file_path = input("Enter the path to the text file: ")
    text = preprocess_text(file_path)
    graph, word_count = build_graph_from_text(text)

    while True:
        print("\nMenu:")
        print("1. Show Directed Graph")
        print("2. Query Bridge Words")
        print("3. Generate New Text")
        print("4. Calculate Shortest Path")
        print("5. Calculate PageRank")
        print("6. Random Walk and Save Path to File")
        print("7. Save Directed Graph as Image")
        print("8. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            show_directed_graph(graph)
        elif choice == '2':
            word1 = input("Enter word1: ")
            word2 = input("Enter word2: ")
            print(query_bridge_words(graph, word1, word2))
        elif choice == '3':
            input_text = input("Enter a line of text: ")
            print(generate_new_text(graph, input_text))
        elif choice == '4':
            word1 = input("Enter word1: ")
            word2 = input("Enter word2 (leave blank if you want all paths from word1): ").strip()
            result = calc_shortest_path(graph, word1, word2 if word2 else None)
            if isinstance(result, tuple):
                path, length = result
                print(f"Shortest path: {path}, Length: {length}")
            else:
                print(result)
        elif choice == '5':
            pagerank = cal_pagerank(graph)
            for word, rank in pagerank.items():
                print(f"{word}: {rank:.4f}")
        elif choice == '6':
            output_file = input("Enter the file path to save the random walk path (e.g., random_walk.txt): ")
            random_walk(graph, output_file)
        elif choice == '7':
            output_file = input("Enter the file path to save the directed graph image (e.g., directed_graph.png): ")
            save_directed_graph_as_image(graph, output_file)
        elif choice == '8':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
