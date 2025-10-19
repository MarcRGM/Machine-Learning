firsts = ["Anna", "Bob", "Charles"]
lasts = ["Smith", "Doe", "Evans"]
for z in zip(firsts, lasts):
     print(z)
# Returns it into tuples
('Anna', 'Smith')
('Bob', 'Doe')
('Charles', 'Evans')


firsts = ["Anna", "Bob", "Charles"]
lasts = ["Smith", "Doe", "Evans"]
for first, last in zip(firsts, lasts):
     print(f"'{first} {last}'")
# Tuple is unpacked during for loop
'Anna Smith'
'Bob Doe'
'Charles Evans'