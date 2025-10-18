#include <iostream>
#include <vector>
#include <algorithm> // For swap

using namespace std;

// Function to maintain the Max Heap property in a subtree
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;       // Assume root is the largest
    int left = 2 * i + 1;  // Left child index
    int right = 2 * i + 2; // Right child index

    // If left child is larger than root
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }

    // If right child is larger than current largest
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }

    // If the largest is not the root, swap and recursively heapify
    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

// Main function to perform Heap Sort
void heapSort(vector<int>& arr) {
    int n = arr.size();

    // 1. Build a Max Heap (rearrange array)
    // Start from the last non-leaf node
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // 2. Extract elements one by one from the heap
    for (int i = n - 1; i > 0; i--) {
        // Move current root (largest element) to the end of the sorted portion
        swap(arr[0], arr[i]);

        // Call heapify on the reduced heap (size 'i')
        heapify(arr, i, 0);
    }
}

// Helper function to print the array
void printArray(const vector<int>& arr) {
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
}

// Driver code to test the Heap Sort implementation
int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7, 2, 4};
    cout << "Original Array: ";
    printArray(arr);

    heapSort(arr);

    cout << "Sorted Array:   ";
    printArray(arr);

    return 0;
}