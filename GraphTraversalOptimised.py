from collections import deque

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]

    def add_edge(self, u, v, directed=False):
        self.adj[u].append(v)
        if not directed:
            self.adj[v].append(u)

    def bfs(self, start):
        visited = [False] * self.V
        queue = deque([start])
        visited[start] = True
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return result

    def dfs_util(self, node, visited, result):
        visited[node] = True
        result.append(node)
        for neighbor in self.adj[node]:
            if not visited[neighbor]:
                self.dfs_util(neighbor, visited, result)

    def dfs(self, start):
        visited = [False] * self.V
        result = []
        self.dfs_util(start, visited, result)
        return result

# Example usage
g = Graph(6)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 4)
g.add_edge(3, 5)
g.add_edge(4, 5)

print("BFS starting from node 0:", g.bfs(0))
print("DFS starting from node 0:", g.dfs(0))
