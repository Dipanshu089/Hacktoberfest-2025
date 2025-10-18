#include <iostream>
#include <vector>
#include <algorithm> // For swap (though not strictly necessary for this implementation's logic)

using namespace std;


void shellSort(vector<int>& arr) {
    int n = arr.size();

    // 1. Outer loop: Determine and shrink the gap size.
    // We start with a large gap and divide it by 2 in each iteration.
    for (int gap = n / 2; gap > 0; gap /= 2) {

        // 2. Middle loop: Perform a gapped Insertion Sort.
        // This is like a regular Insertion Sort, but we look at elements 'gap' distance apart.
        for (int i = gap; i < n; i++) {
            // 'temp' is the element we want to insert into the correct position
            int temp = arr[i];

            int j;
            // 3. Inner loop: Shift elements to make space for the inserted element (temp).
            // Compare arr[j] (current position) with arr[j - gap] (element 'gap' distance behind)
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                // Shift the larger element forward by 'gap' positions
                arr[j] = arr[j - gap];
            }

            // Place 'temp' in its final correct gap-sorted spot
            arr[j] = temp;
        }
    }
}

// Helper function to print the array
void printArray(const vector<int>& arr) {
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
}

// Driver code to test the Shell Sort implementation
int main() {
    vector<int> arr = {12, 34, 54, 2, 3, 1, 9, 8};

    cout << "Original Array: ";
    printArray(arr);

    shellSort(arr);

    cout << "Sorted Array:   ";
    printArray(arr);

    return 0;
}