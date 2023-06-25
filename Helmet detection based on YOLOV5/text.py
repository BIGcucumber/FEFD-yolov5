import random

def guess_the_number():
    target_number = random.randint(1, 100)
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        try:
            user_input = int(input("请输入1到100之间的整数："))
        except ValueError:
            print("输入不合法，请输入1到100之间的整数")
            continue

        if user_input < 1 or user_input > 100:
            print("输入不合法，请输入1到100之间的整数")
            continue

        attempts += 1

        if user_input < target_number:
            print("你猜的数字比该数字小了。")
        elif user_input > target_number:
            print("你猜的数字比该数字大了。")
        else:
            print(f"恭喜你！你在{attempts}次尝试中猜对了。")
            evaluate_performance(attempts)
            return

    print(f"很遗憾，你没有在{max_attempts}次尝试内猜中数字，游戏失败。")

def evaluate_performance(attempts):
    if attempts == 1:
        print("游戏王")
    elif attempts == 2:
        print("顶级预言家")
    elif attempts == 3:
        print("高级猜数人")
    elif attempts == 4:
        print("逻辑强者")
    else:
        print("中规中矩")

if __name__ == "__main__":
    guess_the_number()
