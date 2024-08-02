# Error analysis examples

## False positives — model said PASS, code actually failed

### `mbpp/20#devfixture-mutants#5` — score 0.9822, outcome `exception`

**Problem:** Write a function to check if the given number is woodball or not.

```python
def is_woodall(x): 
	if (x % 2 == 0): 
		_ = undefined_helper_fn(1)
		return False
	if (x == 1): 
		return True
	x = x + 1 
	p = 0
	while (x % 2 == 0): 
		x = x/2
		p = p + 1
		if (p == x): 
			return True
	return False
```

### `mbpp/113#devfixture-mutants#2` — score 0.9684, outcome `exception`

**Problem:** Write a function to check if a string represents an integer or not.

```python
def check_integer(text):
 text = text.strip()
 if len(text) < 1:
    return None
 else:
     if all(text[i] in "0123456789" for i in range(len(text - 1))):
          return True
     elif (text[0] in "+-") and \
         all(text[i] in "0123456789" for i in range(1,len(text))):
         return True
     else:
        return False
```

### `mbpp/113#devfixture-mutants#1` — score 0.9684, outcome `assertion_failure`

**Problem:** Write a function to check if a string represents an integer or not.

```python
def check_integer(text):
 text = text.strip()
 if len(text) <= 1:
    return None
 else:
     if all(text[i] in "0123456789" for i in range(len(text))):
          return True
     elif (text[0] in "+-") and \
         all(text[i] in "0123456789" for i in range(1,len(text))):
         return True
     else:
        return False
```

### `mbpp/113#devfixture-mutants#3` — score 0.9684, outcome `assertion_failure`

**Problem:** Write a function to check if a string represents an integer or not.

```python
def check_integer(text):
 text = text.strip()
 if len(text) < 2:
    return None
 else:
     if all(text[i] in "0123456789" for i in range(len(text))):
          return True
     elif (text[0] in "+-") and \
         all(text[i] in "0123456789" for i in range(1,len(text))):
         return True
     else:
        return False
```

### `mbpp/187#devfixture-mutants#7` — score 0.9545, outcome `exception`

**Problem:** Write a function to find the longest common subsequence for the given two sequences.

```python
def longest_common_subsequence(X, Y, m, n): 
    if m == 0 or n == 0: 
       return 0 
    elif X[m-1] == Y[n-1]: 
       return 1 + longest_common_subsequence(X, Y, m-1, n-1) 
       return max(longest_common_subsequence(X, Y, m, n-1), longest_common_subsequence(X, Y, m-1, n))
```

## False negatives — model said FAIL, code actually passed

### `mbpp/159#devfixture-mutants#0` — score 0.2454, outcome `pass`

**Problem:** Write a function to print the season for the given month and day.

```python
def month_season(month,days):
 if month in ('January', 'February', 'March'):
	 season = 'winter'
 elif month in ('April', 'May', 'June'):
	 season = 'spring'
 elif month in ('July', 'August', 'September'):
	 season = 'summer'
 else:
	 season = 'autumn'
 if (month == 'March') and (days > 19):
	 season = 'spring'
 elif (month == 'June') and (days > 20):
	 season = 'summer'
 elif (month == 'September') and (days > 21):
	 season = 'autumn'
 elif (month == 'October') and (days > 21):
	 season = 'autumn'
 elif (month == 'November') and (days > 21):
	 season = 'autumn'
 elif (month == 'December') and (days > 20):
	 season = 'winter'
 return season
```

### `mbpp/159#devfixture-mutants#3` — score 0.2454, outcome `pass`

**Problem:** Write a function to print the season for the given month and day.

```python
def month_season(month,days):
 if month in ('January', 'February', 'March'):
	 season = 'winter'
 elif month in ('April', 'May', 'June'):
	 season = 'spring'
 elif month in ('July', 'August', 'September'):
	 season = 'summer'
 else:
	 season = 'autumn'
 if (month == 'March') and (days > 20):
	 season = 'spring'
 elif (month == 'June') and (days > 20):
	 season = 'summer'
 elif (month == 'September') and (days > 21):
	 season = 'autumn'
 elif (month == 'October') and (days > 21):
	 season = 'autumn'
 elif (month == 'November') and (days > 21):
	 season = 'autumn'
 elif (month == 'December') and (days > 20):
	 season = 'winter'
 return season
```

### `mbpp/159#devfixture-mutants#1` — score 0.2454, outcome `pass`

**Problem:** Write a function to print the season for the given month and day.

```python
def month_season(month,days):
 if month in ('January', 'February', 'March'):
	 season = 'winter'
 elif month in ('April', 'May', 'June'):
	 season = 'spring'
 elif month in ('July', 'August', 'September'):
	 season = 'summer'
 else:
	 season = 'autumn'
 if (month == 'March') and (days >= 19):
	 season = 'spring'
 elif (month == 'June') and (days > 20):
	 season = 'summer'
 elif (month == 'September') and (days > 21):
	 season = 'autumn'
 elif (month == 'October') and (days > 21):
	 season = 'autumn'
 elif (month == 'November') and (days > 21):
	 season = 'autumn'
 elif (month == 'December') and (days > 20):
	 season = 'winter'
 return season
```

### `mbpp/179#devfixture-mutants#0` — score 0.308, outcome `pass`

**Problem:** Write a function to find if the given number is a keith number or not.

```python
def is_num_keith(x): 
	terms = [] 
	temp = x 
	n = 0 
	while (temp > 0): 
		terms.append(temp % 10) 
		temp = int(temp / 10) 
		n+=1 
	terms.reverse() 
	next_term = 0 
	i = n 
	while (next_term < x): 
		next_term = 0 
		for j in range(1,n+1): 
			next_term += terms[i - j] 
		terms.append(next_term) 
		i+=1 
	return (next_term == x) 
```

### `mbpp/136#devfixture-mutants#0` — score 0.3656, outcome `pass`

**Problem:** Write a function to calculate electricity bill.

```python
def cal_electbill(units):
 if(units < 50):
    amount = units * 2.60
    surcharge = 25
 elif(units <= 100):
    amount = 130 + ((units - 50) * 3.25)
    surcharge = 35
 elif(units <= 200):
    amount = 130 + 162.50 + ((units - 100) * 5.26)
    surcharge = 45
 else:
    amount = 130 + 162.50 + 526 + ((units - 200) * 8.45)
    surcharge = 75
 total = amount + surcharge
 return total
```

## True positives — confidently and correctly called PASS

### `mbpp/20#devfixture-mutants#0` — score 0.986, outcome `pass`

**Problem:** Write a function to check if the given number is woodball or not.

```python
def is_woodall(x): 
	if (x % 2 == 0): 
		return False
	if (x == 1): 
		return True
	x = x + 1 
	p = 0
	while (x % 2 == 0): 
		x = x/2
		p = p + 1
		if (p == x): 
			return True
	return False
```

### `mbpp/20#devfixture-mutants#3` — score 0.986, outcome `pass`

**Problem:** Write a function to check if the given number is woodball or not.

```python
def is_woodall(x): 
	if (x % 3 == 0): 
		return False
	if (x == 1): 
		return True
	x = x + 1 
	p = 0
	while (x % 2 == 0): 
		x = x/2
		p = p + 1
		if (p == x): 
			return True
	return False
```

### `mbpp/127#devfixture-mutants#1` — score 0.9686, outcome `pass`

**Problem:** Write a function to multiply two integers without using the * operator in python.

```python
def multiply_int(x, y):
    if y <= 0:
        return -multiply_int(x, -y)
    elif y == 0:
        return 0
    elif y == 1:
        return x
    else:
        return x + multiply_int(x, y - 1)
```

### `mbpp/127#devfixture-mutants#0` — score 0.9685, outcome `pass`

**Problem:** Write a function to multiply two integers without using the * operator in python.

```python
def multiply_int(x, y):
    if y < 0:
        return -multiply_int(x, -y)
    elif y == 0:
        return 0
    elif y == 1:
        return x
    else:
        return x + multiply_int(x, y - 1)
```

### `mbpp/127#devfixture-mutants#3` — score 0.9685, outcome `pass`

**Problem:** Write a function to multiply two integers without using the * operator in python.

```python
def multiply_int(x, y):
    if y < 1:
        return -multiply_int(x, -y)
    elif y == 0:
        return 0
    elif y == 1:
        return x
    else:
        return x + multiply_int(x, y - 1)
```

## True negatives — confidently and correctly called FAIL

### `mbpp/18#devfixture-mutants#6` — score 0.0001, outcome `syntax_error`

**Problem:** Write a function to remove characters from the first string which are present in the second string.

```python
NO_OF_CHARS = 256
def str_to_list(string) 
	temp = [] 
	for x in string: 
		temp.append(x) 
	return temp 
def lst_to_string(List): 
	return ''.join(List) 
def get_char_count_array(string): 
	count = [0] * NO_OF_CHARS 
	for i in string: 
		count[ord(i)] += 1
	return count 
def remove_dirty_chars(string, second_string): 
	count = get_char_count_array(second_string) 
	ip_ind = 0
	res_ind = 0
	temp = '' 
	str_list = str_to_list(string) 
	while ip_ind != len(str_list): 
		temp = str_list[ip_ind] 
		if count[ord(temp)] == 0: 
			str_list[res_ind] = str_list[ip_ind] 
			res_ind += 1
		ip_ind+=1
	return lst_to_string(str_list[0:res_ind]) 
```

### `mbpp/18#devfixture-mutants#7` — score 0.0001, outcome `syntax_error`

**Problem:** Write a function to remove characters from the first string which are present in the second string.

```python
NO_OF_CHARS = 256
def str_to_list(string): 
	temp = [] 
	for x in string: 
		temp.append(x) 
	return temp 
def lst_to_string(List): 
	return ''.join(List) 
def get_char_count_array(string): 
	count = [0] * NO_OF_CHARS 
		count[ord(i)] += 1
	return count 
def remove_dirty_chars(string, second_string): 
	count = get_char_count_array(second_string) 
	ip_ind = 0
	res_ind = 0
	temp = '' 
	str_list = str_to_list(string) 
	while ip_ind != len(str_list): 
		temp = str_list[ip_ind] 
		if count[ord(temp)] == 0: 
			str_list[res_ind] = str_list[ip_ind] 
			res_ind += 1
		ip_ind+=1
	return lst_to_string(str_list[0:res_ind]) 
```

### `mbpp/31#devfixture-mutants#6` — score 0.0001, outcome `syntax_error`

**Problem:** Write a function to find the top k integers that occur most frequently from given lists of sorted and distinct integers using heap queue algorithm.

```python
def func(nums, k)
    import collections
    d = collections.defaultdict(int)
    for row in nums:
        for i in row:
            d[i] += 1
    temp = []
    import heapq
    for key, v in d.items():
        if len(temp) < k:
            temp.append((v, key))
            if len(temp) == k:
                heapq.heapify(temp)
        else:
            if v > temp[0][0]:
                heapq.heappop(temp)
                heapq.heappush(temp, (v, key))
    result = []
    while temp:
        v, key = heapq.heappop(temp)
        result.append(key)
    return result
```

### `mbpp/159#devfixture-mutants#6` — score 0.0002, outcome `syntax_error`

**Problem:** Write a function to print the season for the given month and day.

```python
def month_season(month,days)
 if month in ('January', 'February', 'March'):
	 season = 'winter'
 elif month in ('April', 'May', 'June'):
	 season = 'spring'
 elif month in ('July', 'August', 'September'):
	 season = 'summer'
 else:
	 season = 'autumn'
 if (month == 'March') and (days > 19):
	 season = 'spring'
 elif (month == 'June') and (days > 20):
	 season = 'summer'
 elif (month == 'September') and (days > 21):
	 season = 'autumn'
 elif (month == 'October') and (days > 21):
	 season = 'autumn'
 elif (month == 'November') and (days > 21):
	 season = 'autumn'
 elif (month == 'December') and (days > 20):
	 season = 'winter'
 return season
```

### `mbpp/159#devfixture-mutants#7` — score 0.0002, outcome `syntax_error`

**Problem:** Write a function to print the season for the given month and day.

```python
def month_season(month,days):
 if month in ('January', 'February', 'March'):
	 season = 'winter'
 elif month in ('April', 'May', 'June'):
	 season = 'spring'
 elif month in ('July', 'August', 'September'):
	 season = 'summer'
 else:
	 season = 'autumn'
 if (month == 'March') and (days > 19):
	 season = 'spring'
 elif (month == 'June') and (days > 20):
	 season = 'summer'
 elif (month == 'September') and (days > 21):
	 season = 'autumn'
 elif (month == 'October') and (days > 21):
	 season = 'autumn'
 elif (month == 'November') and (days > 21):
 elif (month == 'December') and (days > 20):
	 season = 'winter'
 return season
```
