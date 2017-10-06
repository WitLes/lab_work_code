class Solution():
    def reverse(x):
        """
        :type x: int
        :rtype: int
        """
        if x >= 0:
            if x < 10:
                return x
            num = int(str(x)[::-1])
        else:
            num = - int(str(x)[::-1])
        return num *(num < 2**31)

