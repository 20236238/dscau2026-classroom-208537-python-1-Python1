import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================
# 문제 1-1: k차 이동평균 (Moving Average)
# ============================================================

def movingAverages1(X, n, k):
    """
    정의를 그대로 사용하는 버전 (느린 버전) O(nk)
    A[i] = (X[i-k+1] + ... + X[i]) / k   (i >= k-1)
    매 위치마다 k개 원소를 매번 새로 더함
    """
    A = np.zeros(n)
    for i in range(k - 1, n):
        total = 0
        for j in range(i - k + 1, i + 1):
            total += X[j]
        A[i] = total / k
    return A


def movingAverages2(X, n, k):
    """
    중간 합(intermediate sum)을 사용하는 최적화 버전 O(n)
    이전 합에서 맨 앞 원소를 빼고 새 원소를 더함 (슬라이딩 윈도우)
    """
    A = np.zeros(n)
    window_sum = sum(X[:k])
    A[k - 1] = window_sum / k
    for i in range(k, n):
        window_sum += X[i] - X[i - k]
        A[i] = window_sum / k
    return A


# ============================================================
# 문제 1-1a: n=2^5=32, k=2^4=16, 1000번 반복 → 비율 히스토그램
# ============================================================
# ✅ 수정: n=25→32 (2^5), k=24→16 (2^4)
n, k = 32, 16
ratios_1a = []

for _ in range(1000):
    X = np.random.uniform(0, 1, n)

    t0 = time.perf_counter()
    movingAverages1(X, n, k)
    t1 = time.perf_counter()
    movingAverages2(X, n, k)
    t2 = time.perf_counter()

    time1 = t1 - t0
    time2 = t2 - t1

    if time2 > 0:
        ratios_1a.append(time1 / time2)

plt.figure(figsize=(8, 5))
plt.hist(ratios_1a, bins=40, color='steelblue', edgecolor='black')
plt.xlabel('Time Ratio (slow / fast)')
plt.ylabel('Frequency')
plt.title('Histogram of Time Ratios\n(movingAverages1 / movingAverages2), n=2^5, k=2^4')
plt.tight_layout()
plt.savefig('./ratio_hist2.jpg')
plt.close()
print(f"[1-1a] 저장 완료: ratio_hist2.jpg")
print(f"       평균 비율: {np.mean(ratios_1a):.3f}, "
      f"최솟값: {np.min(ratios_1a):.3f}, "
      f"최댓값: {np.max(ratios_1a):.3f}")


# ============================================================
# 문제 1-1b: n=2^4~2^8, k=2^3=8, min/max/avg 비율 플롯
# ============================================================
# ✅ 수정: k=23→8 (2^3)
input_sizes = [2**4, 2**5, 2**6, 2**7, 2**8]
k = 8
min_ratios, max_ratios, avg_ratios = [], [], []

for n in input_sizes:
    ratios_1b = []
    for _ in range(1000):
        X = np.random.uniform(0, 1, n)

        t0 = time.perf_counter()
        movingAverages1(X, n, k)
        t1 = time.perf_counter()
        movingAverages2(X, n, k)
        t2 = time.perf_counter()

        time1 = t1 - t0
        time2 = t2 - t1
        if time2 > 0:
            ratios_1b.append(time1 / time2)

    min_ratios.append(np.min(ratios_1b))
    max_ratios.append(np.max(ratios_1b))
    avg_ratios.append(np.mean(ratios_1b))
    print(f"  n={n:4d} → min={min_ratios[-1]:.3f}, "
          f"avg={avg_ratios[-1]:.3f}, max={max_ratios[-1]:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(input_sizes, min_ratios, 'b-o', label='Min Ratio')
plt.plot(input_sizes, avg_ratios, 'g-o', label='Avg Ratio')
plt.plot(input_sizes, max_ratios, 'r-o', label='Max Ratio')
plt.xlabel('n (input size)')
plt.ylabel('Time Ratio (slow / fast)')
plt.title('Time Ratio vs n (movingAverages1/2), k=2^3')
plt.xticks(input_sizes, [f'$2^{{{i+4}}}$' for i in range(5)])
plt.legend()
plt.tight_layout()
plt.savefig('./ratio_plot2.jpg')
plt.close()
print(f"[1-1b] 저장 완료: ratio_plot2.jpg")


# ============================================================
# 문제 1-2: countOnes — O(n) 알고리즘
# ============================================================

def countOnesButSlow(A, n):
    """
    제공된 느린 버전 O(n^2)
    각 행마다 j=0부터 1을 세기 시작
    """
    c = 0
    for i in range(n):
        j = 0
        while j < n and A[i, j] == 1:
            c += 1
            j += 1
    return c


def countOnes(A, n):
    """
    O(n) 버전 — 오른쪽 위 모서리에서 시작하는 포인터 이동
    핵심 아이디어:
      - 행은 위→아래로 1의 개수가 단조감소
      - (row=0, col=n-1)에서 시작
      - A[row][col]==1 이면 c += col+1 후 row↓
      - A[row][col]==0 이면 col←
      - 총 이동 횟수 ≤ 2n → O(n)
    """
    c = 0
    row = 0
    col = n - 1

    while row < n and col >= 0:
        if A[row, col] == 1:
            c += col + 1
            row += 1
        else:
            col -= 1

    return c


def generate_row_sorted_matrix(n):
    """
    각 행이 1...10...0 형태이고,
    위 행의 1 개수 >= 아래 행의 1 개수인 n×n 행렬 생성
    """
    A = np.zeros((n, n), dtype=int)
    ones_counts = sorted(
        np.random.randint(0, n + 1, size=n), reverse=True
    )
    for i, cnt in enumerate(ones_counts):
        A[i, :cnt] = 1
    return A


# ============================================================
# 문제 1-2a: n=2^6=64, 1000번 → 비율 히스토그램
# ============================================================
n = 2**6   # 64
ratios_2a = []

for _ in range(1000):
    A = generate_row_sorted_matrix(n)

    t0 = time.perf_counter()
    countOnesButSlow(A, n)
    t1 = time.perf_counter()
    countOnes(A, n)
    t2 = time.perf_counter()

    slow_t = t1 - t0
    fast_t = t2 - t1
    if fast_t > 0:
        ratios_2a.append(slow_t / fast_t)

plt.figure(figsize=(8, 5))
plt.hist(ratios_2a, bins=40, color='darkorange', edgecolor='black')
plt.xlabel('Time Ratio (slow / fast)')
plt.ylabel('Frequency')
plt.title('Histogram of Time Ratios\n(countOnesButSlow / countOnes), n=2^6')
plt.tight_layout()
plt.savefig('./ratio_hist4.jpg')
plt.close()
print(f"\n[1-2a] 저장 완료: ratio_hist4.jpg")
print(f"       평균 비율: {np.mean(ratios_2a):.3f}")


# ============================================================
# 문제 1-2b: n=2^4~2^8, min/max/avg 비율 플롯
# ============================================================
input_sizes = [2**4, 2**5, 2**6, 2**7, 2**8]
min_r2, max_r2, avg_r2 = [], [], []

for n in input_sizes:
    ratios_2b = []
    for _ in range(1000):
        A = generate_row_sorted_matrix(n)

        t0 = time.perf_counter()
        countOnesButSlow(A, n)
        t1 = time.perf_counter()
        countOnes(A, n)
        t2 = time.perf_counter()

        slow_t = t1 - t0
        fast_t = t2 - t1
        if fast_t > 0:
            ratios_2b.append(slow_t / fast_t)

    min_r2.append(np.min(ratios_2b))
    max_r2.append(np.max(ratios_2b))
    avg_r2.append(np.mean(ratios_2b))
    print(f"  n={n:4d} → min={min_r2[-1]:.3f}, "
          f"avg={avg_r2[-1]:.3f}, max={max_r2[-1]:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(input_sizes, min_r2, 'b-o', label='Min Ratio')
plt.plot(input_sizes, avg_r2, 'g-o', label='Avg Ratio')
plt.plot(input_sizes, max_r2, 'r-o', label='Max Ratio')
plt.xlabel('n (input size)')
plt.ylabel('Time Ratio (slow / fast)')
plt.title('Time Ratio vs n (countOnesButSlow / countOnes)')
plt.xticks(input_sizes, [f'$2^{{{i+4}}}$' for i in range(5)])
plt.legend()
plt.tight_layout()
plt.savefig('./ratio_plot4.jpg')
plt.close()
print(f"[1-2b] 저장 완료: ratio_plot4.jpg")
