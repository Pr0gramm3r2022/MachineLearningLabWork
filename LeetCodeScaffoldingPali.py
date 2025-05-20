class Solution:
    def isPalindrome(self, x: int) -> bool:
        castx = str(x)
        '''if it reads the same left to right its a palindrome
        iterate through the string left to right and return the values in the cells as a string
        iterate through the string right to left and return the values in the cells as a string
        if the strings are equal, its a palindrome
        typecasting because a string is an array of characters, and 
        can check they are palindromes cell by cell'''

        if x < 0:
            return False
        if x == 0:
            return True
        if x % 10 == 0:
            return False


    OriginalX = x

    numReversed = 0

    while x > 0:
        lastDigit = x % 10
        numreversed = numreversed*10 + lastDigit
        x = x // 10

            return numReversed == Originalx
