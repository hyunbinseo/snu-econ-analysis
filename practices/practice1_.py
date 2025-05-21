# %% [markdown]
# # 경제 분석 및 예측과 데이터 지능 실습1: Getting Started
#
# 본 실습은 파이썬 전반에 대한 기초적인 내용을 아주 간략하게 담고 있습니다.
#
# 처음 파이썬을 접하시는 경우 눈으로 따라오는 것 만으로는 익숙해지기 쉽지 않다고 생각합니다.
#
# 직접 실행도 시켜보고 일부 코드도 바꿔보며 익숙해지시기 바랍니다.
#
# References:
# - [University of Florida Python Basics for Physics](https://cmp.phys.ufl.edu/files/python-tutorial.html)
# - [점프투 파이썬](https://wikidocs.net/book/1)

# %% [markdown]
# ### 목차

# %% [markdown]
# 1. 파이썬 기초
# - Jupyter Notebook에 대한 간단한 소개
# - Type, Operator
# - Control flow
# - Functions
# - Classes
#
# 2. 모듈
# - Numpy
# - Pandas
# - Matplotlib
#
# 3. ARMA 추정 실습

# %% [markdown]
# ## 1. 파이썬 기초

# %% [markdown]
# ### Notebook Editor
#
# 쥬피터 노트북에 대한 간단한 설명

# %% [markdown]
# 쥬피터 노트북의 셀은 CODE, MARKDOWN, RAW의 세 종류를 지정할 수 있습니다. Markdown으로 변경하려면 M을 누르고, CODE로 변경하려면 다시 Y를 누르면 됩니다.
#
# 이 셀은 'MARKDOWN' 셀입니다. 마크다운에 대한 설명은 [여기](https://gist.github.com/ihoneymon/652be052a0727ad59601) 를 참고하세요. ESC를 누르면 마크다운 수정 후 결과를 볼 수 있습니다. SHIFT+ENTER를 눌러도 됩니다.
#
# 아래는 CODE 셀입니다. 마찬가지로 SHIFT+ENTER를 누르면 안에 있는 코드가 실행됩니다.

# %%
print("Hello World!")  # 해시태그 옆에는 주석을 달 수 있습니다.
# print("Hello World!")    이 줄은 실행되지 않습니다.

# %% [markdown]
# 변수로 저장한 값을 텍스트로 아래처럼 출력할 수 도 있습니다.

# %%
s = "Hello World!"  # s 변수를 선언하고 문자열을 할당합니다.
print(f"{s} from Python")  # s는 선언된 변수입니다.

# %% [markdown]
# ### Type, Operator
#
# 변수 타입 및 연산자에 대한 기본 예제

# %% [markdown]
# 파이썬의 변수들은 '포인터' 입니다. JAVA, C++같은 다른 프로그래밍 언어와 달리 데이터 타입을 지정하지 않고도 변수를 선언할 수 있다는 특징이 있습니다.

# %%
x = 17  # x는 정수입니다.
y = 2.3  # y는 부동 소수점 숫자입니다.
z = "hello"  # z는 문자열입니다.
b = True  # b는 논리 연산자 입니다.
l = [1, 2, 3]  # l은 리스트입니다.
t = (4, 5, 6)  # t는 튜플입니다.
d = {"a": 7, "b": 8, "c": 9}  # d는 딕셔너리입니다.
s = {10, 11, 12}  # s는 집합입니다.

# %% [markdown]
# 변수들을 활용한 기초적인 연산은 아래와 같습니다.

# %%
print(f"{x} / 2 = {x / 2}")  # 나눗셈은 기본적으로 부동 소수점 숫자를 반환합니다.
print(f"{x} // 2 = {x // 2}")  # 몫 연산자는 정수를 반환합니다.
print(f"{x} % 2 = {x % 2}")  # 나머지 연산자는 나머지를 반환합니다.
print(f"{x} ** 2 = {x**2}")  # 거듭제곱 연산자는 거듭제곱을 반환합니다.

# %%
print(x != 20)  # x는 20이 아닙니다.
print(x == 20)  # x는 20입니다.
print(x > 20)  # x는 20보다 큽니다.
print(x >= 20)  # x는 20보다 크거나 같습니다.
print(x < 20)  # x는 20보다 작습니다.
print(x <= 20)  # x는 20보다 작거나 같습니다.

# %%
(x > 20) or (x == 17)  # x는 20보다 크거나 17과 같습니다.
(x > 20) and (x == 17)  # x는 20보다 크고 17과 같습니다.
not (x == 17)  # x는 17이 아닙니다.

# %% [markdown]
# 아래는 리스트를 활용한 다양한 연산입니다. 파이썬 언어의 리스트 인덱스는 '0'부터 시작함에 유의하세요

# %%
# 리스트를 출력합니다.
print(l)

# %%
l.append(4)  # 리스트에 4를 추가합니다.
l.extend([5, 6])  # 리스트에 5와 6을 추가합니다.
l.insert(0, 0)  # 리스트의 첫 번째 위치에 0을 추가합니다.
l += [7]  # 리스트에 7을 추가합니다.
l += [8, 9]  # 리스트에 8과 9를 추가합니다.
print(l)
l[0]  # 리스트의 첫 번째 요소를 가져옵니다.
l[1] = 5  # 리스트의 두 번째 요소를 변경합니다.
l[1:3]  # 리스트의 두 번째와 세 번째 요소를 가져옵니다.
l[2:]  # 리스트의 세 번째 요소부터 끝까지를 가져옵니다.
l[:2]  # 리스트의 첫 번째와 두 번째 요소를 가져옵니다.
l[-1]  # 리스트의 마지막 요소를 가져옵니다.
print(l)
l.pop()  # 리스트의 마지막 요소를 제거하고 반환합니다.
l.pop(0)  # 리스트의 첫 번째 요소를 제거하고 반환합니다.
l.index(2)  # 리스트에서 2의 인덱스를 반환합니다.
l.remove(2)  # 리스트에서 2를 제거합니다.
l.count(2)  # 리스트에서 2의 개수를 반환합니다.
l.sort()  # 리스트를 정렬합니다.
l.reverse()  # 리스트를 뒤집습니다.
print(l)
l.clear()  # 리스트를 비웁니다.
n = len(l)  # 리스트의 길이를 반환합니다.
print(f"list {l} has length {n}.")

# 리스트를 재정의 해 줍니다
l = [1, 2, 3]

# %%
l * 3  # 리스트를 세 번 반복합니다.
l + [4, 5, 6]  # 리스트에 4, 5, 6을 추가합니다.
print(l)
print(l + l)  # 리스트끼리 더하면 두 리스트가 합쳐집니다.
(z + " world. ") * 5  # 문자열은 곱셈 연산자를 사용하여 반복할 수 있습니다.
print(list(z))  # 문자열을 리스트로 변환합니다.

# %% [markdown]
# 딕셔너리는 `key`와 `value`의 쌍으로 이루어져 있습니다.

# %%
print(d)
d["a"]  # 딕셔너리에서 'a'의 값을 가져옵니다.
d["d"] = 10  # 딕셔너리에 'd'를 추가합니다.
del d["b"]  # 딕셔너리에서 'b'를 제거합니다.
print(d)
d.keys()  # 딕셔너리의 키를 반환합니다.
d.values()  # 딕셔너리의 값들을 반환합니다.
d.items()  # 딕셔너리의 키와 값들을 반환합니다.
d.get("a")  # 딕셔너리에서 'a'의 값을 가져옵니다.
d.pop("a")  # 딕셔너리에서 'a'를 제거하고 값을 반환합니다.
d.clear()  # 딕셔너리를 비웁니다.
print(d)

# 딕셔너리를 재정의 해 줍니다.
d = {"a": 7, "b": 8, "c": 9}

# %% [markdown]
# 튜플은 함부로 수정할 수 없는 리스트라고 이해하면 좋고 집합(set)은 수학에서 흔하게 활용되는 집합으로 이해하면 됩니다. 집합의 특징은 순서(index)가 없습니다.

# %% [markdown]
# ### Control Flow
#
# 조건문 및 반복문 사용 예제

# %% [markdown]
# `if` 와 `for` 문을 활용한 기초적인 제어문 입니다. 파이썬은 `들여쓰기`로 코드의 범위를 구분 하는 것에 유의해야 합니다.

# %%
x = 7
if x == False:  # x가 False와 같은지 확인합니다.
    print(f"{x} is zero")
elif x > 0:
    print(f"{x} is positive")
elif x in l:  # x가 리스트 l에 포함되어 있는지 확인합니다.
    print(f"{x} belongs to {l}")
else:
    print(f"{x} not found.")

# %% [markdown]
# 아래는 `for` 루프문 입니다. `while` 루프를 활용한 구문도 비슷합니다.

# %%
n = 10  # 몇번 반복할지 정합니다.
for i in range(n):  # range()함수는 0부터 n-1까지의 숫자를 생성합니다.
    print(f"Hello World #{i}")  # 각 반복마다 Hello World를 출력합니다.
    if i > 5:
        break  # 반복문을 종료합니다.
    else:
        continue  # 다음 반복으로 넘어갑니다.
    print("This line will not be printed.")  # 이 줄은 실행되지 않습니다.
else:  # `for`문이 정상적으로 종료되면 실행됩니다.
    print("Finished printing the full list.")  # 이 줄은 실행됩니다.
print("The `for` loop has ended.")  # 이 줄은 실행됩니다.

# %% [markdown]
# `n`을 바꿔가면서 실행해 보세요

# %% [markdown]
# 리스트의 각 element에 대해 반복을 할 수도 있습니다.

# %%
mixed = [x, y, z]  # 변수들로 이루어진 리스트를 생성합니다.
for e in mixed:  # 리스트의 각 요소에 대해 반복합니다.
    print(e, type(e))  # 요소와 요소의 타입을 출력합니다.

# %% [markdown]
# 위의 반복문은 인덱스를 활용하게 되면 아래와 같이 쓸 수도 있습니다.

# %%
for i in range(len(mixed)):
    e = mixed[i]
    print(e, type(e))

# %% [markdown]
# ### Functions

# %% [markdown]
# #### 함수의 개념
#
# 함수는 입력값(인수)을 받아서 특정 작업을 수행한 후 결과(리턴값)를 돌려주는 코드 블록입니다. 마치 믹서에 과일을 넣어 주스를 만드는 것처럼, 함수는 “입력 → 처리 → 출력”의 과정을 거칩니다.
#
# #### 함수를 사용하는 이유
#
# - 코드 재사용 및 유지보수: 반복되는 코드를 하나의 함수로 묶어 여러 번 호출할 수 있습니다.
# - 프로그램 구조화: 프로그램을 기능 단위로 나누어 흐름을 명확히 하고, 디버깅을 쉽게 합니다.
#
# #### 파이썬 함수의 기본 구조


# %%
def 함수명(매개변수):
    # 수행할 문장들
    return 리턴값


# %% [markdown]
# 매개변수는 함수가 입력받는 값이며, 인수를 통해 실제 값을 전달받습니다.
#
# #### 함수의 다양한 형태


# %%
# 일반 함수: 입력값과 리턴값이 모두 있는 함수
def add(a, b):
    return a + b


# 입력값이 없는 함수
def say():
    return "Hi"


# 리턴값이 없는 함수
def add_no_return(a, b):
    print(f"{a} + {b} = {a + b}")


# 입력값도 리턴값도 없는 함수
def say_no_return():
    print("Hi")


# *args를 사용하면 입력값의 개수가 정해지지 않은 경우 튜플로 받아 처리할 수 있습니다.
def add_many(*args):
    result = 0
    for num in args:
        result += num
    return result


# **kwargs를 사용하면 키워드 인자들을 딕셔너리 형태로 받아 처리할 수 있습니다.
def print_kwargs(**kwargs):
    print(kwargs)


# lambda 함수

add_lambda = lambda a, b: a + b
print(add_lambda(3, 4))  # 출력: 7


# %% [markdown]
# 함수는 항상 하나의 리턴값을 돌려줍니다. 여러 값을 반환할 경우 튜플로 반환됩니다.
#
# return 문을 만나면 즉시 함수 실행이 종료되며, 이후 코드는 실행되지 않습니다.
#
# 또한 함수 내부에서 정의된 매개변수 및 변수는 함수 외부와 독립적입니다.

# %%
simple_add = add(1, 2)  # 1과 2를 더합니다.
print(f"simple_add: {simple_add}")
simple_say = say()  # Hi를 출력합니다.
print(f"simple_say: {simple_say}")
add_no_return(1, 2)  # 1과 2를 더한 값을 출력합니다.
say_no_return()  # Hi를 출력합니다.
add_many_result = add_many(
    1, 2, 3, 4, 5
)  # 1부터 5까지 더한 값을 반환합니다(출력 안됨).
print(f"add_many_result: {add_many_result}")
print_kwargs(a=1, b=2, c=3)  # a, b, c를 출력합니다.

# %% [markdown]
# ### Classes

# %% [markdown]
# #### 클래스의 정의

# %% [markdown]
# - `클래스(Class)`는 객체를 생성하기 위한 설계도(일종의 과자 틀)입니다.
# - `객체(Object) 또는 인스턴스(Instance)`는 클래스로부터 만들어진 실체이며, 각각 독립적인 속성과 동작(메서드)을 가집니다.
#
# #### 클래스가 필요한 이유
# - 코드 재사용 및 유지보수: 여러 객체에서 공통 기능을 정의하여 반복 코드를 줄이고, 수정 시 한 곳만 수정하면 됩니다.
# - 데이터와 기능의 결합: 관련 데이터(속성)와 데이터를 처리하는 기능(메서드)을 하나의 단위로 묶어 프로그램 구조를 명확히 합니다.
# - 확장성: 상속(Inheritance)과 메서드 오버라이딩을 통해 기존 클래스를 확장하거나 수정하여 다양한 기능을 손쉽게 추가할 수 있습니다.

# %%
# 사칙연산 기능을 가진 계산기 클래스를 정의합니다.


class FourCal:
    # 생성자: 객체 생성 시 자동 호출되어 객체변수를 초기화합니다.
    def __init__(self, first, second):
        self.first = first
        self.second = second

    # 두 수의 합을 리턴하는 메서드
    def add(self):
        return self.first + self.second

    # 두 수의 차를 리턴하는 메서드
    def sub(self):
        return self.first - self.second

    # 두 수의 곱을 리턴하는 메서드
    def mul(self):
        return self.first * self.second

    # 두 수의 나눗셈을 리턴하는 메서드 (0으로 나눌 경우 0을 리턴)
    def div(self):
        if self.second == 0:
            return 0
        return self.first / self.second


# %%
# 객체 생성 및 메서드 사용 예제
a = FourCal(4, 2)
print(a.add())  # 출력: 6
print(a.mul())  # 출력: 8

# %% [markdown]
# #### 클래스와 객체의 관계
#
# - 클래스는 객체를 만들기 위한 설계도이며, 한 클래스에서 여러 객체(인스턴스)를 생성할 수 있습니다.
# - 각 객체는 클래스에서 정의한 속성(예: a.first, a.second)과 메서드(add, sub 등)를 가지며, 서로 독립적으로 동작합니다.
#
# #### 상속과 메서드 오버라이딩
# - 상속(Inheritance): 기존 클래스를 확장하여 새로운 기능을 추가할 수 있습니다.
# - 메서드 오버라이딩: 상속받은 클래스에서 부모 클래스의 메서드를 같은 이름으로 재정의하여, 필요에 따라 기능을 변경할 수 있습니다.
# - 예를 들어, FourCal 클래스를 상속받아 거듭제곱 기능을 추가한 MoreFourCal 클래스를 만들어 보겠습니다.


# %%
class MoreFourCal(FourCal):
    def pow(self):
        return self.first**self.second


b = MoreFourCal(4, 2)
print(b.pow())  # 출력: 16

# %% [markdown]
# #### 클래스 변수
# 클래스 변수는 클래스 내에 정의된 변수로, 해당 클래스로 생성된 모든 객체가 공유합니다.


# %%
# Family 클래스: 1세대 가족을 표현합니다.
class Family:
    # 클래스 변수: 모든 Family 객체가 공유하는 국적 (초기값: South Korean)
    nationality = "South Korean"

    def __init__(self, parents=None, kids=None):
        # 인스턴스 변수: 각 Family 객체별로 고유한 부모, 자녀 목록을 저장합니다.
        if parents is None:
            parents = []
        if kids is None:
            kids = []
        self.parents = parents
        self.kids = kids

    def show_members(self):
        """가족 구성원과 현재 국적을 출력합니다."""
        print("Parents:", self.parents)
        print("Kids:", self.kids)
        print("Nationality:", self.nationality)

    def change_nationality(self, new_nationality):
        """클래스 변수를 변경하여 모든 가족의 국적을 변경합니다."""
        Family.nationality = new_nationality
        print(f"Nationality changed to {new_nationality} for all families!")

    def add_kid(self, kid_name):
        """자녀를 추가합니다."""
        self.kids.append(kid_name)


# NextGenerationFamily 클래스: Family 클래스를 상속받아 2세대 가족을 표현합니다.
# 조부모 정보는 부모 세대 Family 객체들의 부모 목록을 결합하여 자동으로 도출됩니다.
class NextGenerationFamily(Family):
    def __init__(self, parents=None, kids=None, parent_families=None):
        # 부모 클래스의 __init__을 호출하여 parents와 kids를 초기화합니다.
        super().__init__(parents, kids)
        # 조부모 목록 초기화
        self.grandparents = []
        # parent_families: 1세대 가족 객체들의 리스트를 전달받아 조부모를 도출합니다.
        if parent_families is not None:
            for fam in parent_families:
                self.grandparents.extend(fam.parents)

    # 부모 클래스의 show_members()를 오버라이딩하여 조부모 정보도 함께 출력합니다.
    def show_members(self):
        super().show_members()
        print("Grandparents:", self.grandparents)


# %% [markdown]
# #### '가족'클래스를 활용한 설명
#
# Generation 1:
# - Family1: Tom과 Susan, 자녀: Michael, John
# - Family2: Josh와 Sarah, 자녀: Maddison, Conner
#
# Generation 2:
# - Michael와 Maddison이 결혼하여 자녀 Yuna를 둡니다.
#   → 조부모: Tom, Susan, Josh, Sarah
#
# 초기 국적은 "South Korean"이며, 이후 "Canadian"으로 변경된다고 가정

# %%
# Generation 1 가족 객체 생성
family1 = Family(parents=["Tom", "Susan"], kids=["Michael", "John"])
family2 = Family(parents=["Josh", "Sarah"], kids=["Maddison", "Conner"])

print("=== Generation 1 Families ===")
print("Family 1:")
family1.show_members()
print("\nFamily 2:")
family2.show_members()

# %%
# Generation 2 가족 객체 생성
# Michael (family1의 자녀)와 Maddison (family2의 자녀)이 결혼하여 Yuna라는 자녀를 둡니다.
family3 = NextGenerationFamily(
    parents=["Michael", "Maddison"], kids=["Yuna"], parent_families=[family1, family2]
)

print("\n=== Generation 2 Family ===")
family3.show_members()

# %%
# 국적 변경: 한 가족의 국적만 "Canadian"으로 변경.
print("\n=== Changing Nationality to Canadian ===")
family3.nationality = "Canadian"

print("\n=== Nationality After Change ===")
print("Family 1:")
family1.show_members()
print("\nFamily 2:")
family2.show_members()
print("\nFamily 3:")
family3.show_members()

# %%
# 국적 변경: 모든 가족의 국적이 "Canadian"으로 변경
print("\n=== Changing Nationality to Canadian ===")
family3.change_nationality("Canadian")

print("\n=== Nationality After Change ===")
print("Family 1:")
family1.show_members()
print("\nFamily 2:")
family2.show_members()
print("\nFamily 3:")
family3.show_members()

# %% [markdown]
# ## 2. 모듈, 패키지, 라이브러리

# %% [markdown]
# 모듈은 함수, 클래스, 변수 등이 정의되어 있는 파일로, 한 번 작성한 코드를 여러 프로그램에서 재사용할 수 있게 해줍니다.
#
# #### 모듈의 장점
# - 재사용성: 여러 프로젝트나 파일에서 동일한 기능을 사용하고 싶을 때, 모듈을 import하여 중복 코드를 줄일 수 있습니다.
# - 조직화: 관련 있는 코드들을 한 파일에 모아두면 관리와 유지보수가 쉬워집니다.
#
# #### 모듈 사용법
# - import 문: 모듈 전체를 불러올 때는 import 모듈명을 사용합니다.
# - 긴 모듈 이름을 짧게 사용하고 싶을 때는 as 키워드를 사용해 별칭을 부여할 수 있습니다.
#
# 이러한 모듈을 모아놓은 것을 패키지라고 하며, 패키지들을 모아놓은 것을 라이브러리라고 합니다.
#
# 통상적으로 하나의 파일을 모듈이라고 하고, 여러 파일들로 이루어 진 것을 패키지라고 합니다.
#
# 라이브러리 >= 패키지 >= 모듈

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ### Numpy

# %% [markdown]
# - NumPy는 Python에서 고성능 수치 계산을 위한 핵심 라이브러리입니다.
# - NumPy는 다차원 배열 객체(ndarray)를 제공하며, 벡터화 연산, 선형대수, 통계, 그리고 다양한 수학 함수들을 효율적으로 수행할 수 있습니다.
#
# #### 배열 생성
# NumPy 배열은 Python 리스트를 사용하거나, np.arange(), np.linspace(), np.zeros(), np.ones()와 같은 함수를 통해 생성할 수 있습니다.

# %%
# 리스트를 배열로 변환하기
a = np.array([1, 2, 3, 4, 5])
print("a =", a)

# np.arange()를 사용하여 배열 생성: 0부터 10 미만까지 2씩 증가하는 배열
b = np.arange(0, 10, 2)
print("b =", b)

# np.linspace()를 사용하여 배열 생성: 0부터 1까지 5개의 숫자를 균등 분포
c = np.linspace(0, 1, 5)
print("c =", c)

# np.zeros()와 np.ones()로 배열 생성
zeros = np.zeros((2, 3))
ones = np.ones((2, 3))
print("zeros =\n", zeros)
print("ones =\n", ones)

# %% [markdown]
# #### 배열의 속성과 기본 연산
#
# 배열은 shape, dtype 등 다양한 속성을 가지며, 벡터화 연산을 지원하여 반복문 없이도 빠른 계산이 가능합니다.

# %%
print("a의 shape:", a.shape)
print("a의 데이터 타입:", a.dtype)

# 배열 간의 기본 산술 연산 (요소별 연산)
x = np.array([10, 20, 30, 40, 50])
print("a + x =", a + x)  # 요소별 덧셈
print("a * 2 =", a * 2)  # 각 요소에 2를 곱함

# 스칼라와 배열 간의 연산 (Broadcasting)
print("a + 100 =", a + 100)

# %% [markdown]
# #### 다차원 배열 및 인덱싱
#
# NumPy는 다차원 배열을 쉽게 다룰 수 있으며, 인덱싱과 슬라이싱을 통해 원하는 요소에 접근할 수 있습니다.

# %%
# 2차원 배열 생성
d = np.array([[1, 2, 3], [4, 5, 6]])
print("d =\n", d)
print("d의 shape:", d.shape)

# 인덱싱 및 슬라이싱
print("d[0, 1] =", d[0, 1])

# 첫 번째 행, 두 번째 열
print("d[:, 1] =", d[:, 1])
# 모든 행의 두 번째 열
print("d[1, :] =", d[1, :])
# 두 번째 행 전체
print("d[0:2, 1:3] =\n", d[0:2, 1:3])

# %% [markdown]
# #### 배열의 브로드캐스팅, 집계함수
# - 브로드캐스팅은 서로 다른 크기의 배열 간에도 연산이 가능하도록 자동으로 크기를 맞춰주는 기능입니다.
# - NumPy는 배열의 합계, 평균, 최대값, 최소값 등 다양한 통계 함수를 제공합니다.

# %%
# 1차원 배열과 2차원 배열 간의 연산 예제
e = np.array([[1, 2, 3], [4, 5, 6]])
f = np.array([10, 20, 30])
print("e + f =\n", e + f)  # f가 각 행에 자동으로 브로드캐스팅됨

print("d의 합계:", np.sum(d))
print("d의 평균:", np.mean(d))
print("d의 최대값:", np.max(d))
print("d의 최소값:", np.min(d))
print("d의 표준편차:", np.std(d))

# %% [markdown]
# #### 행렬 연산 및 기타 기능
#
# - 두 배열의 행렬 곱은 np.dot() 또는 @ 연산자를 사용하여 계산할 수 있습니다.
# - 배열의 모양을 바꾸거나 차원을 확장/축소하는 작업을 할 수도 있습니다.

# %%
# 1차원 배열의 내적
dot_product = np.dot(a, x)
print("a와 x의 dot product =", dot_product)

# 2차원 배열의 행렬 곱
g = np.array([[1, 2], [3, 4]])
h = np.array([[5, 6], [7, 8]])
print("g =\n", g)
print("h =\n", h)
print("g.dot(h) =\n", np.dot(g, h))
print("g @ h =\n", g @ h)

# 배열 재구성: reshape()를 사용하여 배열의 모양 변경
i = np.arange(12)  # 0부터 11까지의 1차원 배열
print("i =", i)
j = i.reshape((3, 4))  # 3행 4열로 재구성
print("j =\n", j)

# 배열 평탄화: flatten()을 사용하면 다차원 배열을 1차원으로 변환
k = j.flatten()
print("k =", k)

# 0과 1 사이의 균등 분포 난수 5개 생성
rand_uniform = np.random.rand(5)
print("균등 분포 난수:", rand_uniform)

# 표준 정규분포 난수 (평균 0, 표준편차 1) 생성 (3x3 배열)
rand_normal = np.random.randn(3, 3)
print("정규분포 난수 (3x3):\n", rand_normal)

# %% [markdown]
# ### Pandas
#
# Pandas는 데이터 조작과 분석을 위한 강력한 라이브러리입니다.
# 주요 데이터 구조로는 **Series**(1차원 배열)와 **DataFrame**(2차원 표 형식 데이터)가 있습니다.
#
# #### Series와 Dataframe
# - Series는 1차원 배열과 같이 작동하며, 인덱스가 자동으로 할당되거나 사용자 지정할 수 있습니다.
# - DataFrame은 여러 열로 구성된 2차원 데이터 구조입니다. 리스트 혹은 딕셔너리를 사용하여 쉽게 생성할 수 있습니다.

# %%
# 리스트를 사용하여 Series 생성
data = [10, 20, 30, 40, 50]
s = pd.Series(data)
print("Default Index Series:\n", s)

# 사용자 지정 인덱스를 사용하여 Series 생성
s2 = pd.Series(data, index=["a", "b", "c", "d", "e"])
print("\nCustom Index Series:\n", s2)

# 리스트를 사용하여 DataFrame 생성
data = [
    ["Alice", 25, 50000],
    ["Bob", 30, 60000],
    ["Charlie", 35, 70000],
    ["David", 40, 80000],
]
df = pd.DataFrame(data, columns=["Name", "Age", "Salary"])
print("DataFrame:\n", df)

# 딕셔너리를 사용하여 Series 생성
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 40],
    "Salary": [50000, 60000, 70000, 80000],
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# %% [markdown]
# #### 인덱싱 및 데이터 선택
#
# DataFrame의 열은 딕셔너리 키와 비슷하게 선택할 수 있으며, `.loc`와 `.iloc`를 통해 행을 선택할 수 있습니다.

# %%
# 열 선택
print("Name 열:\n", df["Name"])

# .loc를 이용한 행 선택 (인덱스 라벨)
print("\n두 번째 행 (loc):\n", df.loc[1])

# .iloc를 이용한 행 선택 (정수 위치)
print("\n세 번째 행 (iloc):\n", df.iloc[2])

# 슬라이싱 예제
print("\n0~2번째 행 (loc):\n", df.loc[0:2])

# %% [markdown]
# #### 기본 DataFrame 연산 및 집계 함수
#
# Pandas는 데이터 요약과 통계 함수를 제공합니다.

# %%
# 데이터 요약 통계
print("요약 통계:\n", df.describe())

# 특정 열로 정렬
df_sorted = df.sort_values(by="Age")
print("\n나이 순 정렬:\n", df_sorted)

# 조건에 따른 필터링
df_filtered = df[df["Salary"] > 60000]
print("\nSalary가 60000 초과인 행:\n", df_filtered)

# %% [markdown]
# #### 파일 입출력
#
# Pandas는 CSV, Excel, SQL 등 다양한 포맷의 파일을 쉽게 읽고 쓸 수 있습니다.

# %%
# CSV 파일 읽기
df_from_csv = pd.read_csv("../datasets/RSCCASN.csv")
print(df_from_csv)

# DataFrame을 CSV 파일로 저장하기 (인덱스는 저장하지 않음)
# df.to_csv('output.csv', index=False)

# %% [markdown]
# #### 결측치 처리
#
# Pandas는 `fillna`, `dropna` 등의 함수로 결측치를 다룹니다.

# %%
# 결측치가 있는 DataFrame 생성
data_with_nan = {"A": [1, 2, None, 4], "B": [None, 2, 3, 4]}
df_nan = pd.DataFrame(data_with_nan)
print("원본 데이터 (결측치 포함):\n", df_nan)

# 결측치를 특정 값으로 채우기
df_filled = df_nan.fillna(0)
print("\n결측치 채우기 (0으로):\n", df_filled)

# 결측치가 있는 행 제거
df_dropped = df_nan.dropna()
print("\n결측치가 있는 행 제거:\n", df_dropped)

# %% [markdown]
# #### 데이터 병합 혹은 연걸
#
# 두 데이터프레임을 같은 값을 기준으로 병합하거나 그냥 쌓아서 연결할 수 있습니다.

# %%
# 두 DataFrame 병합 (inner join)
df1 = pd.DataFrame({"key": ["A", "B", "C", "D"], "value1": [1, 2, 3, 4]})
df2 = pd.DataFrame({"key": ["B", "C", "D", "E"], "value2": [5, 6, 7, 8]})
merged_df = pd.merge(df1, df2, on="key", how="inner")
print("병합된 DataFrame:\n", merged_df)

# DataFrame 연결 (세로로 연결)
concat_df = pd.concat([df1, df2], axis=0, sort=False)
print("\n연결된 DataFrame:\n", concat_df)


# %% [markdown]
# ### Matplotlib

# %% [markdown]
# Matplotlib은 Python에서 데이터 시각화를 위한 대표적인 라이브러리입니다.
# 주로 `pyplot` 모듈을 사용하여 간단한 코드로 다양한 그래프(선 그래프, 산점도, 히스토그램 등)를 그릴 수 있습니다.

# %%
# 선 그래프 예제
x = np.linspace(0, 10, 100)  # 0부터 10까지 100개의 숫자 생성
y = np.sin(x)  # sin 함수를 적용한 값

plt.figure()  # 새로운 Figure 생성
plt.plot(x, y, label="sin(x)")  # x, y 데이터를 선 그래프로 그리기
plt.xlabel("x value")  # x축 라벨 설정
plt.ylabel("sin(x)")  # y축 라벨 설정
plt.title("sin function graph")  # 그래프 제목 설정
plt.legend()  # 범례 표시
plt.show()  # 그래프 출력

# %%
# 산점도 예제
x_scatter = np.random.rand(50)  # 0과 1 사이의 랜덤값 50개 생성
y_scatter = np.random.rand(50)  # 0과 1 사이의 랜덤값 50개 생성

plt.figure()
plt.scatter(x_scatter, y_scatter, color="green", marker="o")
plt.xlabel("x value")
plt.ylabel("y value")
plt.title("scatter plot example")
plt.show()

# %%
# 히스토그램 예제
data = np.random.randn(1000)  # 평균 0, 표준편차 1인 1000개의 난수 생성

plt.figure()
plt.hist(data, bins=30, color="skyblue", edgecolor="black")
plt.xlabel("value")
plt.ylabel("frequency")
plt.title("histogram example")
plt.show()

# %% [markdown]
# ## 3. ARMA 추정 실습

# %% [markdown]
# #### 데이터를 준비합니다.
#
# 본 실습에서는 AEP_hourly 데이터를 활용하며, ARMA(2,1) 모형을 추정합니다.
#
# $y_{t} = \phi_{1} y_{t-1} + \phi_{2} y_{t-2} + \epsilon_{t} + \psi \epsilon_{t-1}, \quad \epsilon_{t} \sim N(0, \sigma^{2})$

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 데이터 로드
df = pd.read_csv("../datasets/AEP_hourly.csv", parse_dates=["Datetime"])
df = df.sort_values(by="Datetime").reset_index(drop=True)
print(df.head())

# 실습을 위해 일부 데이터만 사용합니다.
df_subset = df.head(2000)
y = df_subset["AEP_MW"].values
T = len(y)
print("Number of observations:", T)

# %%
from scipy.optimize import minimize


# ARMA(2,1) 모델의 음의 로그 우도 함수를 정의합니다.
# 모델: X_t = phi1 * X_{t-1} + phi2 * X_{t-2} + epsilon_t + psi * epsilon_{t-1}
# 여기서 epsilon_t ~ N(0, sigma2)
def arma_loglike(params, y):  # 파라미터를 추출합니다: phi1, phi2, psi, sigma2
    phi1, phi2, psi, sigma2 = params
    T = len(y)
    # 잔차를 초기화하고 처음 두 개의 잔차를 0으로 설정합니다.
    eps = np.zeros(T)
    # t=2부터 T-1까지 잔차를 재귀적으로 계산합니다.
    # 참고: t=0과 t=1에 대해서는 eps = 0으로 가정합니다.
    for t in range(2, T):
        eps[t] = y[t] - phi1 * y[t - 1] - phi2 * y[t - 2] - psi * eps[t - 1]
    # t=2부터 T-1까지의 잔차를 사용합니다.
    n = T - 2
    # 조건부 로그 우도를 계산합니다 (초기 조건은 무시합니다)
    # 로그 우도 = -0.5 * n * log(2πσ^2) - (1/(2σ^2)) * sum(eps[t]^2)
    ll = -0.5 * n * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * np.sum(eps[2:] ** 2)
    # 대부분의 최적화 알고리즘은 함수를 최소화하므로 음의 로그 우도를 반환합니다.
    return -ll


# %% [markdown]
# ARMA(2,1) 모델의 최적화를 위한 초기 추정값을 설정합니다.
# - 일반적으로 AR 패러미터는 OLS를 사용하여 초기 추정값을 설정하며, MA 패러미터는 0으로 설정합니다.
# - sigma2의 초기 추정값은 데이터의 분산으로 설정합니다.

# %%
# ARMA(2,1) 모델의 초기 추정값을 계산합니다.
phi_ols = np.linalg.inv(
    np.column_stack((y[1:-1], y[0:-2])).T @ np.column_stack((y[1:-1], y[0:-2]))
) @ (np.column_stack((y[1:-1], y[0:-2])).T @ y[2:])
init_phi1, init_phi2 = phi_ols
init_psi = 0.0
init_sigma2 = np.var(y)
init_params = np.array([init_phi1, init_phi2, init_psi, init_sigma2])

# phi1, phi2, psi는 (-∞, ∞) 범위, sigma2는 (0, ∞) 범위로 설정합니다.
bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (1e-6, np.inf)]

# Optimize the negative log likelihood.
result = minimize(
    arma_loglike, init_params, args=(y,), bounds=bounds, method="L-BFGS-B"
)
est_phi1, est_phi2, est_psi, est_sigma2 = result.x

print("Estimated ARMA(2,1) parameters:")
print(f"  phi1  = {est_phi1:.4f}")
print(f"  phi2  = {est_phi2:.4f}")
print(f"  psi = {est_psi:.4f}")
print(f"  sigma^2 = {est_sigma2:.4f}")


# %%
plt.figure(figsize=(10, 5))
plt.plot(np.arange(T), y, linestyle="-", color="skyblue", label="AEP_MW")
plt.title("AEP_MW Time Series")
plt.xlabel("Time Index")
plt.ylabel("AEP_MW")
plt.grid(True)
plt.legend(["AEP_MW"])
plt.show()

# %% [markdown]
# ### APPENDIX: 추정 과정에 대한 보다 자세한 설명은 아래와 같습니다
#
# 모델: ARMA(2,1)
# $$
# X_t = \phi_1\,X_{t-1} + \phi_2\,X_{t-2} + \epsilon_t + \psi\,\epsilon_{t-1}, \quad t = 3,4,\dots,T.
# $$
#
# % OLS를 활용한 AR(2) 추정(초기값)
# $$
# Y = \begin{pmatrix}
# X_3 \\
# X_4 \\
# \vdots \\
# X_T
# \end{pmatrix}, \qquad
# Z = \begin{pmatrix}
# X_2 & X_1 \\
# X_3 & X_2 \\
# \vdots & \vdots \\
# X_{T-1} & X_{T-2}
# \end{pmatrix}.
# $$
#
# $$
# \hat{\boldsymbol{\phi}} = \begin{pmatrix}\hat{\phi}_1 \\ \hat{\phi}_2\end{pmatrix} = (Z^\top Z)^{-1} Z^\top Y.
# $$
#
# 잔차 초기값
# $$
# \epsilon_1 = 0, \quad \epsilon_2 = 0.
# $$
#
# 잔차를 구하기 위해 초기값 대입
# $$
# Y_t^* = X_t - \phi_1\,X_{t-1} - \phi_2\,X_{t-2}, \quad t \ge 3.
# $$
#
# 잔차를 구함
# $$
# \epsilon_t = Y_t^* - \psi\,\epsilon_{t-1}, \quad t \ge 3.
# $$
#
# 구한 잔차와 초기값으로 매트릭스 구성
# $$
# Y^* = \begin{pmatrix}
# Y_3^* \\
# Y_4^* \\
# \vdots \\
# Y_T^*
# \end{pmatrix}, \qquad
# \boldsymbol{\epsilon} = \begin{pmatrix}
# \epsilon_3 \\
# \epsilon_4 \\
# \vdots \\
# \epsilon_T
# \end{pmatrix}.
# $$
#
# 인덱스 이동 매트릭스(M) 적용
# $$
# M\,\boldsymbol{\epsilon} = \begin{pmatrix}
# 0 \\
# \epsilon_3 \\
# \epsilon_4 \\
# \vdots \\
# \epsilon_{T-1}
# \end{pmatrix}.
# $$
#
# 매트릭스로 표현한 MA 부분
# $$
# \Bigl(I + \psi\,M\Bigr)\,\boldsymbol{\epsilon} = Y^*,
# $$
#
# $$
# \boldsymbol{\epsilon} = \Bigl(I + \psi\,M\Bigr)^{-1}Y^*.
# $$
#
# 로그 우도 함수 구성
# $$
# \ell(\phi_1,\phi_2,\psi,\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=3}^{T}\epsilon_t^2, \quad n = T-2.
# $$
#
# 매트릭스로 표현한 잔차합
# $$
# \sum_{t=3}^{T}\epsilon_t^2 = \boldsymbol{\epsilon}^\top\boldsymbol{\epsilon} = Y^{*T}\Bigl[(I+\psi\,M)^{-T}(I+\psi\,M)^{-1}\Bigr]Y^*.
# $$
#
# 매트릭스로 표현한 우도 함수
# $$
# \ell(\phi_1,\phi_2,\psi,\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\,Y^{*T}\Bigl[(I+\psi\,M)^{-T}(I+\psi\,M)^{-1}\Bigr]Y^*.
# $$
#
# 패러미터 업데이트(H는 헤시안, s는 스코어를 의미하며 2차미분, 1차 미분을 뜻함)
# $$
# \theta^{(k+1)} = \theta^{(k)} - H^{-1}\bigl(\theta^{(k)}\bigr)\, s\bigl(\theta^{(k)}\bigr),
# $$
#
# $$
# \theta = (\phi_1,\phi_2,\psi,\sigma^2)^\top,
# $$
# $$
# s(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}, \qquad H(\theta) = \frac{\partial^2 \ell(\theta)}{\partial \theta\,\partial \theta^\top}.
# $$
#
# 이렇게 해서 수렴한 $\theta$를 통해 최종 패러미터를 구할 수 있습니다.

# %%
