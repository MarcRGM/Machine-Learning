# SET comprehension
# Create a set of squares from 1 to 5
squares = {x**2 for x in range(1, 6)}
print(squares)

# DICTIONARY comprehension
# Create a dictionary where each number maps to its square
square_dict = {x: x**2 for x in range(1, 6)}
print(square_dict)



# Combination of SET and DICTIONARY comprehension
word2id = {w: i + 1 for i, w in enumerate(['I', 'write', 'Python'])}
# is the same as
word2id1 = {}
for i, w in enumerate(['I', 'write', 'Python']):
    word2id1[w] = i + 1

print(word2id)
print(word2id1)
