import numpy as np
import matplotlib.pyplot as plt
import time
import array

# 랜덤 시드 고정 (결과 재현성 확보)
np.random.seed(42)

# ============================================================
# 문제 1-1: k차 이동평균 (Moving Average)
# ============================================================

def movingAverages1(X, n, k):
    A = np.zeros(n)
    for i in range(k - 1, n):
        total = 0
        for j in range(i - k + 1, i + 1):
            total += X[j]
        A[i] = total / k
    return A

def movingAverages2(X, n, k):
    A = np.zeros(n)
    window_sum = sum(X[:k])
    A[k - 1] = window_sum / k
    for i in range(k, n):
        window_sum += X[i] - X[i - k]
        A[i] = window_sum / k
    return A

# --- 문제 1-1a ---
n, k = 32, 16
ratios_1a = []

for _ in range(1000):
    X = np.random.uniform(0, 1, n)
    t0 = time.perf_counter(); movingAverages1(X, n, k); t1 = time.perf_counter()
    t2 = time.perf_counter(); movingAverages2(X, n, k); t3 = time.perf_counter()
    time1, time2 = t1 - t0, t3 - t2
    if time2 > 0: ratios_1a.append(time1 / time2)

plt.figure(figsize=(8, 5))
plt.hist(ratios_1a, bins=40, color='steelblue', edgecolor='black')
plt.xlabel('Time Ratio (slow / fast)'); plt.ylabel('Frequency')
plt.title('Histogram of Time Ratios\n(movingAverages1 / movingAverages2), n=2^5, k=2^4')
plt.tight_layout(); plt.savefig('./ratio_hist2.jpg'); plt.close()
print(f"[1-1a] 저장 완료: ratio_hist2.jpg")

# --- 문제 1-1b ---
input_sizes = [2**4, 2**5, 2**6, 2**7, 2**8]
k = 8
min_ratios, max_ratios, avg_ratios = [], [], []

for n in input_sizes:
    ratios_1b = []
    for _ in range(1000):
        X = np.random.uniform(0, 1, n)
        t0 = time.perf_counter(); movingAverages1(X, n, k); t1 = time.perf_counter()
        t2 = time.perf_counter(); movingAverages2(X, n, k); t3 = time.perf_counter()
        time1, time2 = t1 - t0, t3 - t2
        if time2 > 0: ratios_1b.append(time1 / time2)
    min_ratios.append(np.min(ratios_1b)); max_ratios.append(np.max(ratios_1b)); avg_ratios.append(np.mean(ratios_1b))

plt.figure(figsize=(8, 5))
plt.plot(input_sizes, min_ratios, 'b-o', label='Min Ratio')
plt.plot(input_sizes, avg_ratios, 'g-o', label='Avg Ratio')
plt.plot(input_sizes, max_ratios, 'r-o', label='Max Ratio')
plt.xlabel('n (input size)'); plt.ylabel('Time Ratio (slow / fast)')
plt.title('Time Ratio vs n (movingAverages1/2), k=2^3')
plt.xticks(input_sizes, [f'$2^{{{i+4}}}$' for i in range(5)])
plt.legend(); plt.tight_layout(); plt.savefig('./ratio_plot2.jpg'); plt.close()
print(f"[1-1b] 저장 완료: ratio_plot2.jpg")


# ============================================================
# 문제 1-2: countOnes
# ============================================================

def countOnesButSlow(A, n):
    c = 0
    for i in range(n):
        j = 0
        while j < n and A[i, j] == 1:
            c += 1; j += 1
    return c

def countOnes(A, n):
    c, row, col = 0, 0, n - 1
    while row < n and col >= 0:
        if A[row, col] == 1: c += col + 1; row += 1
        else: col -= 1
    return c

def generate_row_sorted_matrix(n):
    A = np.zeros((n, n), dtype=int)
    ones_counts = sorted(np.random.randint(0, n + 1, size=n), reverse=True)
    for i, cnt in enumerate(ones_counts): A[i, :cnt] = 1
    return A

# --- 문제 1-2a ---
n = 2**6
ratios_2a = []
for _ in range(1000):
    A = generate_row_sorted_matrix(n)
    t0 = time.perf_counter(); countOnesButSlow(A, n); t1 = time.perf_counter()
    t2 = time.perf_counter(); countOnes(A, n); t3 = time.perf_counter()
    if (t3 - t2) > 0: ratios_2a.append((t1 - t0) / (t3 - t2))

plt.figure(figsize=(8, 5))
plt.hist(ratios_2a, bins=40, color='darkorange', edgecolor='black')
plt.xlabel('Time Ratio (slow / fast)'); plt.ylabel('Frequency')
plt.title('Histogram of Time Ratios\n(countOnesButSlow / countOnes), n=2^6')
plt.tight_layout(); plt.savefig('./ratio_hist4.jpg'); plt.close()
print(f"\n[1-2a] 저장 완료: ratio_hist4.jpg")

# --- 문제 1-2b ---
input_sizes = [2**4, 2**5, 2**6, 2**7, 2**8]
min_r2, max_r2, avg_r2 = [], [], []
for n in input_sizes:
    ratios_2b = []
    for _ in range(1000):
        A = generate_row_sorted_matrix(n)
        t0 = time.perf_counter(); countOnesButSlow(A, n); t1 = time.perf_counter()
        t2 = time.perf_counter(); countOnes(A, n); t3 = time.perf_counter()
        if (t3 - t2) > 0: ratios_2b.append((t1 - t0) / (t3 - t2))
    min_r2.append(np.min(ratios_2b)); max_r2.append(np.max(ratios_2b)); avg_r2.append(np.mean(ratios_2b))

plt.figure(figsize=(8, 5))
plt.plot(input_sizes, min_r2, 'b-o', label='Min Ratio')
plt.plot(input_sizes, avg_r2, 'g-o', label='Avg Ratio')
plt.plot(input_sizes, max_r2, 'r-o', label='Max Ratio')
plt.xlabel('n (input size)'); plt.ylabel('Time Ratio (slow / fast)')
plt.title('Time Ratio vs n (countOnesButSlow / countOnes)')
plt.xticks(input_sizes, [f'$2^{{{i+4}}}$' for i in range(5)])
plt.legend(); plt.tight_layout(); plt.savefig('./ratio_plot4.jpg'); plt.close()
print(f"[1-2b] 저장 완료: ratio_plot4.jpg")


# ============================================================
# 문제 2: 재귀 (Recursion) 
# ============================================================

def gcd1(a, b):
    print(f"  computing gcd1({a}, {b})")  
    if b == 0: return a
    return gcd1(b, a % b)

def gcd2(a, b):
    print(f"  computing gcd2({a}, {b})")  
    if a == b: return a
    if a > b: return gcd2(a - b, b)
    else: return gcd2(a, b - a)

print("\n[2-1a] gcd(493, 33)")
res_gcd1_a = gcd1(493, 33)
print(" -----")
res_gcd2_a = gcd2(493, 33)
print(f"  -> Result: gcd1 is {res_gcd1_a}, gcd2 is {res_gcd2_a}")

print("\n[2-1b] gcd(225, 13)")
res_gcd1_b = gcd1(225, 13)
print(" -----")
res_gcd2_b = gcd2(225, 13)
print(f"  -> Result: gcd1 is {res_gcd1_b}, gcd2 is {res_gcd2_b}")

def divide(a, b):
    print(f"  computing divide({a}, {b})")  
    if a < b: return (0, a)
    q, r = divide(a - b, b)
    return (q + 1, r)

print("\n[2-2a] divide(413, 31)")
res_div_a = divide(413, 31)
print(f"  -> Result: quotient={res_div_a[0]}, remainder={res_div_a[1]}")

print("\n[2-2b] divide(325, 113)")
res_div_b = divide(325, 113)
print(f"  -> Result: quotient={res_div_b[0]}, remainder={res_div_b[1]}")


# ============================================================
# 문제 3: Basic Data Structure — Sparse Matrix
# ============================================================

def run_sparse_experiment_optimized(num_ones_range, n_trials=10**6):
    ratios = np.zeros(n_trials)
    for i in range(n_trials):
        k = np.random.randint(num_ones_range[0], num_ones_range[1] + 1)
        flat_indices = np.random.choice(100, size=k, replace=False)
        ratios[i] = np.sum(flat_indices) / (k * (2 * k - 1))
    return ratios

print("\n[3-1] 실험 중 (1~5개 1s)...")
ratios1 = run_sparse_experiment_optimized((1, 5))
plt.figure(figsize=(8, 6))
plt.hist(ratios1, bins=50, color='royalblue', edgecolor='black')
plt.title('Ratio of Offsets\n(1~5 ones)')
plt.xlabel('Offset Ratio (Dense / Sparse)'); plt.ylabel('Frequency')
plt.tight_layout(); plt.savefig('./sparse_hist1.jpg'); plt.close()
print("[3-1] sparse_hist1.jpg 저장 완료")

print("\n[3-1] 실험 중 (10~20개 1s)...")
ratios2 = run_sparse_experiment_optimized((10, 20))
plt.figure(figsize=(8, 6))
plt.hist(ratios2, bins=50, color='tomato', edgecolor='black')
plt.title('Ratio of Offsets\n(10~20 ones)')
plt.xlabel('Offset Ratio (Dense / Sparse)'); plt.ylabel('Frequency')
plt.tight_layout(); plt.savefig('./sparse_hist2.jpg'); plt.close()
print("[3-1] sparse_hist2.jpg 저장 완료")


# ============================================================
# 문제 4: Lists
# ============================================================

def delete_middle_list(lst):
    del lst[len(lst) // 2]

def delete_middle_array(arr):
    del arr[len(arr) // 2]

# ------------------------------------------------------------
# 4.1 (a) & (b): Deletion Time Histograms
# ------------------------------------------------------------
n = 100
times_list_del = []
times_array_del = []

for _ in range(1000):
    lst = list(range(n))
    t0 = time.perf_counter(); delete_middle_list(lst); t1 = time.perf_counter()
    times_list_del.append(t1 - t0)
    
    arr = array.array('i', range(n))
    t2 = time.perf_counter(); delete_middle_array(arr); t3 = time.perf_counter()
    times_array_del.append(t3 - t2)

# 4.1 (a) 저장
plt.figure(figsize=(8, 5))
plt.hist(times_list_del, bins=40, color='steelblue', edgecolor='black')
plt.xlabel('Elapsed Time (s)'); plt.ylabel('Frequency')
plt.title('Problem 4.1 (a): Histogram of List Deletion Time (n=100)')
plt.tight_layout(); plt.savefig('./histListDel.jpg'); plt.close()
print("\n[Problem 4.1 (a)] histListDel.jpg 저장 완료")

# 4.1 (b) 저장
plt.figure(figsize=(8, 5))
plt.hist(times_array_del, bins=40, color='darkorange', edgecolor='black')
plt.xlabel('Elapsed Time (s)'); plt.ylabel('Frequency')
plt.title('Problem 4.1 (b): Histogram of Array Deletion Time (n=100)')
plt.tight_layout(); plt.savefig('./histArrayDel.jpg'); plt.close()
print("[Problem 4.1 (b)] histArrayDel.jpg 저장 완료")

# ------------------------------------------------------------
# 4.1 (c): List vs Array Deletion Ratio
# ------------------------------------------------------------
sizes_4 = [100, 300, 500, 700, 900]
min_r4, avg_r4, max_r4 = [], [], []

for n_size in sizes_4:
    ratios = []
    for _ in range(1000):
        lst = list(range(n_size))
        t0 = time.perf_counter(); delete_middle_list(lst); t1 = time.perf_counter()
        
        arr = array.array('i', range(n_size))
        t2 = time.perf_counter(); delete_middle_array(arr); t3 = time.perf_counter()
        
        if (t3 - t2) > 0 and (t1 - t0) > 0:
            ratios.append((t1 - t0) / (t3 - t2))
            
    min_r4.append(np.min(ratios)); avg_r4.append(np.mean(ratios)); max_r4.append(np.max(ratios))

plt.figure(figsize=(8, 5))
plt.plot(sizes_4, min_r4, 'b-o', label='Min')
plt.plot(sizes_4, avg_r4, 'g-o', label='Avg')
plt.plot(sizes_4, max_r4, 'r-o', label='Max')
plt.xlabel('n (Elements)'); plt.ylabel('Ratio (List Time / Array Time)')
plt.title('Problem 4.1 (c): List vs Array Deletion Time Ratio')
plt.xticks(sizes_4); plt.legend(); plt.tight_layout()
plt.savefig('./ListVsArrayDel.jpg'); plt.close()
print("[Problem 4.1 (c)] ListVsArrayDel.jpg 저장 완료")


# ------------------------------------------------------------
# 4.2 (Sharing)
# ------------------------------------------------------------
np.random.seed(0)
A_share = np.random.binomial(1, 0.2, size=(20, 10))

# 4.2 (a)
def SharingList(A):
    n_elements, n_groups = A.shape
    result = [[] for _ in range(n_groups)]
    for g in range(n_groups):
        for e in range(n_elements):
            if A[e, g] == 1:
                result[g].append(e)
    return result

print("\n[Problem 4.2 (a)] SharingList(A) 출력 결과:\n", SharingList(A_share))

# 4.2 (b)
def FindPopularList(A):
    sharing = SharingList(A)
    n_elements = A.shape[0]
    counts = [0] * n_elements
    for group in sharing:
        for e in group: counts[e] += 1
    return counts.index(max(counts))

def FindPopularArray(A):
    counts = np.sum(A, axis=1)
    return int(np.argmax(counts))

print(f"  [확인용] FindPopularList  결과: {FindPopularList(A_share)}")
print(f"  [확인용] FindPopularArray 결과: {FindPopularArray(A_share)}")

times_list_sh, times_array_sh = [], []
for _ in range(1000):
    t0 = time.perf_counter(); FindPopularList(A_share); t1 = time.perf_counter()
    t2 = time.perf_counter(); FindPopularArray(A_share); t3 = time.perf_counter()
    times_list_sh.append(t1 - t0)
    times_array_sh.append(t3 - t2)

ratios_sh = [l/a for l, a in zip(times_list_sh, times_array_sh) if a > 0]
plt.figure(figsize=(8, 5))
plt.hist(ratios_sh, bins=40, color='mediumpurple', edgecolor='black')
plt.xlabel('Ratio (List Time / Array Time)'); plt.ylabel('Frequency')
plt.title('Problem 4.2 (b): Histogram of FindPopular Time Ratio\n(List / Array)')
plt.tight_layout(); plt.savefig('./HistSharing.jpg'); plt.close()
print("[Problem 4.2 (b)] HistSharing.jpg 저장 완료")


# ============================================================
# 문제 5: Sets
# ============================================================

class Node:
    def __init__(self, elem):
        self.elem = elem
        self.next = None

class OrderedSet:
    def __init__(self):
        self.head = None
    def add(self, e):
        new_node = Node(e)
        if self.head is None or self.head.elem > e:
            new_node.next = self.head; self.head = new_node; return
        cur = self.head
        while cur.next and cur.next.elem < e: cur = cur.next
        if cur.next and cur.next.elem == e: return
        new_node.next = cur.next; cur.next = new_node
    def member(self, e):
        if self.head is None: return False
        p = self.head
        while True:
            a = p.elem
            if a < e:
                if p.next is None: return False
                p = p.next
            elif a > e: return False
            else: return True
    def is_empty(self):
        return self.head is None

def Subset(A, B):
    if A.is_empty(): return True
    p = A.head
    while True:
        if B.member(p.elem):
            if p.next is None: return True
            p = p.next
        else: return False

def SubsetFast(A, B):
    if A.is_empty(): return True
    p_a, p_b = A.head, B.head
    while p_a is not None:
        if p_b is None: return False
        if p_b.elem < p_a.elem: p_b = p_b.next
        elif p_b.elem == p_a.elem: p_a, p_b = p_a.next, p_b.next
        else: return False
    return True

def make_ordered_set(elements):
    s = OrderedSet()
    for e in elements: s.add(e)
    return s

A_set = make_ordered_set([0, 9])
sizes_5 = [10, 30, 50, 70, 90]
min_r5, avg_r5, max_r5 = [], [], []

for n_size in sizes_5:
    ratios = []
    for _ in range(1000):
        elems = np.random.choice(101, size=n_size, replace=False)
        B_set = make_ordered_set(elems.tolist())
        t0 = time.perf_counter(); Subset(A_set, B_set); t1 = time.perf_counter()
        t2 = time.perf_counter(); SubsetFast(A_set, B_set); t3 = time.perf_counter()
        if (t3 - t2) > 0: ratios.append((t1 - t0) / (t3 - t2))
    min_r5.append(np.min(ratios)); avg_r5.append(np.mean(ratios)); max_r5.append(np.max(ratios))

plt.figure(figsize=(8, 5))
plt.plot(sizes_5, min_r5, 'b-o', label='Min')
plt.plot(sizes_5, avg_r5, 'g-o', label='Avg')
plt.plot(sizes_5, max_r5, 'r-o', label='Max')
plt.xlabel('n (Size of B)'); plt.ylabel('Time Ratio (Subset / SubsetFast)')
plt.title('Problem 5: Subset vs SubsetFast Time Ratio\n(A = {0, 9})')
plt.xticks(sizes_5); plt.legend(); plt.tight_layout()
plt.savefig('./Subset.jpg'); plt.close()
print("\n[Problem 5] Subset.jpg 저장 완료")

print("\n===== 모든 문제 출력 완료 =====")
