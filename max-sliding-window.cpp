/*
 * Function to find the maximum sliding window
 * Using multiset to maintain the current window elements
 * Time Complexity: O(n log k)
 */
#include <iostream>
#include <vector>
#include <set>
using namespace std;
vector<int> maxSlidingWindow(vector<int> &nums, int k) {
        multiset<int> s;
        vector<int> ret;
        for (int i = 0; i < k; i++) { s.insert(nums[i]); }
        for (int i = k; i < nums.size(); i++) {
                ret.push_back(*s.rbegin());
                s.erase(s.find(nums[i - k]));
                s.insert(nums[i]);
        }
        ret.push_back(*s.rbegin());
        return ret;
}