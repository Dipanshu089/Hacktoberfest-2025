"""
Topological Sort Implementation

Topological Sort (Topo Sort) is a linear ordering of vertices in a Directed Acyclic Graph (DAG).
In this ordering, for every directed edge u → v, vertex u comes before vertex v.

Key Properties:
- Works only for Directed Acyclic Graphs (DAGs)
- Cannot have cycles - if the graph has a cycle, topological sorting is not possible
- Multiple valid orders may exist for the same graph

Applications:
- Task scheduling (e.g., course prerequisites, build systems)
- Dependency resolution
- Job scheduling with dependencies

This module implements two approaches:
1. DFS-based approach: Uses depth-first search with a stack
2. Kahn's Algorithm: BFS-based approach using in-degree calculation
"""

from collections import deque

graph = []
visited = []
stack = []

def topological_sort_dfs(node):
    """DFS-based topological sort helper function"""
    visited[node] = True
    
    for neighbor in range(len(graph)):
        if graph[node][neighbor] == 1 and not visited[neighbor]:
            topological_sort_dfs(neighbor)
    
    stack.append(node)

def topological_sort_util():
    """
    Performs topological sort using DFS approach.
    Returns the topologically sorted order of vertices.
    Works only for Directed Acyclic Graphs (DAG).
    """
    global visited, stack
    visited = [False] * len(graph)
    stack = []
    
    for node in range(len(graph)):
        if not visited[node]:
            topological_sort_dfs(node)
    
    return stack[::-1]

def kahns_algorithm():
    """
    Performs topological sort using Kahn's Algorithm (BFS approach).
    Returns the topologically sorted order of vertices.
    Works only for Directed Acyclic Graphs (DAG).
    Also detects if the graph has a cycle.
    """
    num_nodes = len(graph)
    in_degree = [0] * num_nodes
    
    # Calculate in-degree for each vertex
    for u in range(num_nodes):
        for v in range(num_nodes):
            if graph[u][v] == 1:
                in_degree[v] += 1
    
    # Queue for vertices with in-degree 0
    queue = deque()
    for i in range(num_nodes):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        
        # Reduce in-degree for neighbors
        for neighbor in range(num_nodes):
            if graph[node][neighbor] == 1:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    # Check if topological sort is possible (no cycle)
    if len(topo_order) != num_nodes:
        print("Graph has a cycle! Topological sort not possible.")
        return []
    
    return topo_order

def init_graph(num_nodes):
    """
    Initialize a Directed Acyclic Graph (DAG) for topological sort.
    Example: A course prerequisite graph where edges represent dependencies.
    """
    global graph
    graph = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Creating a DAG representing course prerequisites
    # Course 0 -> Course 1 (Course 0 must be taken before Course 1)
    # Course 0 -> Course 2
    # Course 1 -> Course 3
    # Course 2 -> Course 3
    # Course 2 -> Course 4
    # Course 3 -> Course 4
    graph[0][1] = 1
    graph[0][2] = 1
    graph[1][3] = 1
    graph[2][3] = 1
    graph[2][4] = 1
    graph[3][4] = 1

if __name__ == "__main__":
    init_graph(5)
    
    print("Topological Sort using DFS:")
    result_dfs = topological_sort_util()
    print(" -> ".join(map(str, result_dfs)))
    
    print("\nTopological Sort using Kahn's Algorithm (BFS):")
    result_kahn = kahns_algorithm()
    print(" -> ".join(map(str, result_kahn)))
