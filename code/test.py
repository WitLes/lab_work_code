class Solution():
    def maxSubArray(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        curSum = maxSum = nums[0]
        for num in nums[1:]:
            curSum = max(num, curSum + num)
            maxSum = max(maxSum, curSum)

        return maxSum

a = [-2,1,-3,4,-1,2,1,-5,4]
sum_a = Solution.maxSubArray(a)
print(sum_a)