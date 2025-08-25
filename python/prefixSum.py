def runningSum(nums):
    result = [nums[0]]

    for i in range (1, len(nums)):
        result.append(nums[i-1] + nums[i])

    return result

numbers = [1, 2, 3, 4, 5]
result = runningSum(numbers)

print(result)
