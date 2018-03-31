# 阶乘的实现
def factorial(x):
	if x == 1:
		return 1
	else:
		return x*factorial(x-1)

def loop_factorial(x):
	factorial = 1
	while x:
		factorial = factorial * x
		x -= 1
	return factorial
#print(factorial(5))
#print(loop_factorial(4))


# 乘方的实现,还可以用power(),y = x**n
def loop_power(x):
	product = 1
	for n in range(1,x+1):
		product = 365*product
	return product
#print(loop_power(2))
#print('*',365**2)


for x in range(1,100):
	numerator = (loop_factorial(365)/
		         loop_factorial(365-x))
	denominator = loop_power(x)
	propability = numerator/denominator
	if propability<=0.5:
		print('method1:',x) 
		break

x = propability = 1
while propability>0.5:
	numerator = loop_factorial(365)/loop_factorial(365-x)
	denominator = loop_power(x)
	propability = numerator/denominator
	x += 1
print('method2:',x-1)