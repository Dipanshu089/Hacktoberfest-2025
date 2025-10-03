//Grid Game Solution (Java) – Prefix Sum Approach
class Solution {
    public long gridGame(int[][] grid) {
        int n = grid[0].length;

        long[] topPrefixSum = new long[n+1];
        long[] bottomPrefixSum = new long[n+1];

        for(int i=0; i<n; i++)
        {
            topPrefixSum[i+1] = topPrefixSum[i] + grid[0][i];
            bottomPrefixSum[i+1] = bottomPrefixSum[i] + grid[1][i];
        }
        long minMaxPoint = Long.MAX_VALUE;

        for(int c = 0; c<n; c++)
        {
            long pointTop = topPrefixSum[n] - topPrefixSum[c+1];
            long pointBottom = bottomPrefixSum[c];

            long maxPoint = Math.max(pointTop , pointBottom);
            minMaxPoint = Math.min(minMaxPoint, maxPoint);
        }
        return minMaxPoint;
    }
}
