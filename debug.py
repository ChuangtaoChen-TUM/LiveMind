from typing import Generator

def echo_msg() -> Generator[int, str|None, None]:
    num = 0
    while num < 10:
        msg = yield num
        if msg == "STOP":
            yield 0
            break
        num += 1


if __name__ == "__main__":
    echo = echo_msg()
    for i in echo:
        print(i)
        if i == 5:
            echo.send("STOP")
            break