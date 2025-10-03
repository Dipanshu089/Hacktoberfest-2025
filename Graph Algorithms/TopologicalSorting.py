#include <bits/stdc++.h>
using namespace std;

// Function to perform DFS and topological sorting
void topologicalSortUtil(int v, vector<vector<int>> &adj, vector<bool> &visited, stack<int> &st)
{

    // Mark the current node as visited
    visited[v] = true;

    // Recur for all adjacent vertices
    for (int i : adj[v])
    {
        if (!visited[i])
            topologicalSortUtil(i, adj, visited, st);
    }

    // Push current vertex to stack which stores the result
    st.push(v);
}

vector<vector<int>> constructadj(int V, vector<vector<int>> &edges)
{

    vector<vector<int>> adj(V);
    for (auto it : edges)
    {
        adj[it[0]].push_back(it[1]);
    }

    return adj;
}

// Function to perform Topological Sort
vector<int> topologicalSort(int V, vector<vector<int>> &edges)
{

    // Stack to store the result
    stack<int> st;

    vector<bool> visited(V, false);
    vector<vector<int>> adj = constructadj(V, edges);
    // Call the recursive helper function to store
    // Topological Sort starting from all vertices one by
    // one
    for (int i = 0; i < V; i++)
    {
        if (!visited[i])
            topologicalSortUtil(i, adj, visited, st);
    }

    vector<int> ans;

    // Append contents of stack
    while (!st.empty())
    {
        ans.push_back(st.top());
        st.pop();
    }

    return ans;
}

int main()
{

    // Graph represented as an adjacency list
    int v = 6;
    vector<vector<int>> edges = {{2, 3}, {3, 1}, {4, 0}, {4, 1}, {5, 0}, {5, 2}};

    vector<int> ans = topologicalSort(v, edges);

    for (auto node : ans)
    {
        cout << node << " ";
    }
    cout << endl;

    return 0;
}
