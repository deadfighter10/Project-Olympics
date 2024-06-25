prime = []


def fact(n):
    factors = []
    for i in range(1, int(n**(1/2))+1):
        if n % i == 0:
            factors.append(i)
            factors.append(int(n/i))
    factors.remove(n)
    return factors

num = int(input("number: "))
ans = []

for i in range(1, num+1):
    # check if the number is prime
    if fact(i) == [] or fact(i) == [1]:
        prime.append(i)
        #print(f"prime: {i}")
    else:
        #print(f"not prime: {i}")
        pass

    if sum(fact(i)) == i:
        ans.append(i)
        print(f"perfect: {i} <----------------------------")

print(ans)
