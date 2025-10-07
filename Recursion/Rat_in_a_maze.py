
def printAllPaths(maze, n):
    directions = ['D', 'L', 'R', 'U']
    row_change = [1, 0, 0, -1]
    col_change = [0, -1, 1, 0]

    def isSafe(x, y, visited):
        return (0 <= x < n and 0 <= y < n and maze[x][y] == 1 and not visited[x][y])

    def solve(x, y, path, visited, result):
        if x == n - 1 and y == n - 1:
            result.append(path)
            return

        visited[x][y] = True

        for i in range(4):
            new_x = x + row_change[i]
            new_y = y + col_change[i]
            if isSafe(new_x, new_y, visited):
                solve(new_x, new_y, path + directions[i], visited, result)

        visited[x][y] = False

    result = []
    visited = [[False for _ in range(n)] for _ in range(n)]

    if maze[0][0] == 1:
        solve(0, 0, "", visited, result)

    if not result:
        print("No path found!")
    else:
        print("All possible paths:")
        for p in result:
            print(p)


# Example Maze (1 = open path, 0 = blocked)
maze = [
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 1]
]

n = len(maze)
printAllPaths(maze, n)
