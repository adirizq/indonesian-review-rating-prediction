import matplotlib.pyplot as plt

total = [6915, 2175, 4345, 10591, 82491]
label = ['Rating 1: 6,915', 'Rating 2: 2,175', 'Rating 3: 4,345', 'Rating 4: 10,591', 'Rating 5: 82,491']

print(6915 + 2175 + 4345 + 10591 + 82491)


plt.pie(total, labels=label, autopct='%1.1f%%', pctdistance=0.85)
plt.title('Label Distribution')
plt.show()

total_train = [4978, 1568, 3092, 7600, 59455]
total_valid = [549, 173, 362, 891, 6546]
total_test = [1388, 434, 891, 2100, 16490]

label_train = ['Rating 1: 4,978', 'Rating 2: 1,568', 'Rating 3: 3,092', 'Rating 4: 7,600', 'Rating 5: 59,455']
label_valid = ['Rating 1: 549', 'Rating 2: 173', 'Rating 3: 362', 'Rating 4: 891', 'Rating 5: 6,546']
label_test = ['Rating 1: 1,388', 'Rating 2: 434', 'Rating 3: 891', 'Rating 4: 2,100', 'Rating 5: 16,490']


print(4978 + 1568 + 3092 + 7600 + 59455)

plt.pie(total_train, labels=label_train, autopct='%1.1f%%')
plt.title('Train Dataset Label Distribution')
plt.show()


print(549 + 173 + 362 + 891 + 6546)

plt.pie(total_valid, labels=label_valid, autopct='%1.1f%%')
plt.title('Validation Dataset Label Distribution')
plt.show()


print(1388 + 434 + 891 + 2100 + 16490)

plt.pie(total_test, labels=label_test, autopct='%1.1f%%')
plt.title('Test Dataset Label Distribution')
plt.show()
