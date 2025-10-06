# Implementation of Johnson's algorithm in Python3

# Import function to initialize the dictionary
from collections import defaultdict
INT_MAX = float('Inf')

# Function that returns the vertex 
# with minimum distance 
# from the source
def Min_Distance(dist, visit):

    (minimum, Minimum_Vertex) = (INT_MAX, 0)
    for vertex in range(len(dist)):
        if minimum > dist[vertex] and visit[vertex] == False:
            (minimum, minVertex) = (dist[vertex], vertex)

    return Minimum_Vertex


# Dijkstra Algorithm for Modified
# Graph (After removing the negative weights)
def Dijkstra_Algorithm(graph, Altered_Graph, source):

    # Number of vertices in the graph
    tot_vertices = len(graph)

    # Dictionary to check if given vertex is
    # already included in the shortest path tree
    sptSet = defaultdict(lambda : False)

    # Shortest distance of all vertices from the source
    dist = [INT_MAX] * tot_vertices

    dist[source] = 0

    for count in range(tot_vertices):

        # The current vertex which is at min Distance
        # from the source and not yet included in the
        # shortest path tree
        curVertex = Min_Distance(dist, sptSet)
        sptSet[curVertex] = True

        for vertex in range(tot_vertices):
            if ((sptSet[vertex] == False) and
                (dist[vertex] > (dist[curVertex] +
                Altered_Graph[curVertex][vertex])) and
                (graph[curVertex][vertex] != 0)):
                                 
                                                    dist[vertex] = (dist[curVertex] +Altered_Graph[curVertex][vertex])

    # Print the Shortest distance from the source
    for vertex in range(tot_vertices):
        print ('Vertex ' + str(vertex) + ': ' + str(dist[vertex]))

# Function to calculate shortest distances from source
# to all other vertices using Bellman-Ford algorithm
def BellmanFord_Algorithm(edges, graph, tot_vertices):

    # Add a source s and calculate its min
    # distance from every other node
    dist = [INT_MAX] * (tot_vertices + 1)
    dist[tot_vertices] = 0

    for i in range(tot_vertices):
        edges.append([tot_vertices, i, 0])

    for i in range(tot_vertices):
        for (source, destn, weight) in edges:
            if((dist[source] != INT_MAX) and
                    (dist[source] + weight < dist[destn])):
                dist[destn] = dist[source] + weight

    # Don't send the value for the source added
    return dist[0:tot_vertices]

# Function to implement Johnson Algorithm
def JohnsonAlgorithm(graph):

    edges = []

    # Create a list of edges for Bellman-Ford Algorithm
    for i in range(len(graph)):
        for j in range(len(graph[i])):

            if graph[i][j] != 0:
                edges.append([i, j, graph[i][j]])

    # Weights used to modify the original weights
    Alter_weigts = BellmanFord_Algorithm(edges, graph, len(graph))

    Altered_Graph = [[0 for p in range(len(graph))] for q in
                    range(len(graph))]

    # Modify the weights to get rid of negative weights
    for i in range(len(graph)):
        for j in range(len(graph[i])):

            if graph[i][j] != 0:
                Altered_Graph[i][j] = (graph[i][j] +
                        Alter_weigts[i] - Alter_weigts[j]);

    print ('Modified Graph: ' + str(Altered_Graph))

    # Run Dijkstra for every vertex as source one by one
    for source in range(len(graph)):
        print ('\nShortest Distance with vertex ' +
                        str(source) + ' as the source:\n')
        Dijkstra_Algorithm(graph, Altered_Graph, source)

# Driver Code
graph = [[0, -5, 2, 3],
        [0, 0, 4, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]]

JohnsonAlgorithm(graph)
