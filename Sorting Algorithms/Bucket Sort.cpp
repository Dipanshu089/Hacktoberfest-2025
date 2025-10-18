#include <iostream>
#include <vector>
#include <algorithm> // For std::sort
#include <cmath>     // For floor

using namespace std;

// Main function to perform Bucket Sort
void bucketSort(vector<float>& arr) {
    int n = arr.size();
    if (n <= 1) return;

    // 1. Create 'n' empty buckets (a vector of vectors)
    vector<vector<float>> buckets(n);

    // 2. Scatter: Place each element into a specific bucket
    for (float val : arr) {
        // Calculate the index for the bucket based on the value (assuming 0.0 to 1.0 range)
        int bucketIndex = floor(n * val);
        buckets[bucketIndex].push_back(val);
    }

    // 3. Sort: Sort the individual elements inside each bucket
    for (int i = 0; i < n; i++) {
        // We use C++'s built-in sort (which is highly optimized, usually IntroSort)
        // because the buckets are expected to be small.
        sort(buckets[i].begin(), buckets[i].end());
    }

    // 4. Gather: Concatenate all sorted buckets back into the original array
    int index = 0;
    for (const auto& bucket : buckets) {
        for (float val : bucket) {
            arr[index++] = val;
        }
    }
}

// Helper function to print the array
void printArray(const vector<float>& arr) {
    for (float x : arr) {
        // Print with 2 decimal places for neatness
        printf("%.2f ", x);
    }
    cout << endl;
}

// Driver code to test the Bucket Sort implementation
int main() {
    // Note: The input data should ideally be between 0.0 and 1.0
    vector<float> arr = {0.89f, 0.45f, 0.68f, 0.12f, 0.90f, 0.29f, 0.75f, 0.33f};

    cout << "Original Array: ";
    printArray(arr);

    bucketSort(arr);

    cout << "Sorted Array:   ";
    printArray(arr);

    return 0;
}