"""Terminal input helpers (no business logic)."""
from __future__ import annotations


class Prompter:
    def int_range(self, lower: int, upper: int, msg: str = "Enter selection: ") -> int:
        correct = False
        value = lower
        while not correct:
            raw = input(msg)
            if len(raw) == 0:
                print("Invalid input: No input")
                continue
            try:
                value = int(raw)
            except ValueError:
                print("Invalid input: ValueError. Input must be a number e.g. 1")
                continue
            if value in range(lower, upper + 1):
                correct = True
            else:
                print("Input out of bounds: Input must be between " + str(lower) + " and " + str(upper))
        return value

    def float_range(self, lower: float, upper: float, msg: str = "Enter selection: ") -> float:
        correct = False
        value = lower
        while not correct:
            raw = input(msg)
            if len(raw) == 0:
                print("Invalid input: No input")
                continue
            try:
                value = float(raw)
            except ValueError:
                print("Invalid input: ValueError. Input must be a number e.g. 0.5")
                continue
            if value <= upper and value >= lower:
                correct = True
            else:
                print("Input out of bounds: Input must be between " + str(lower) + " and " + str(upper))
        return value

    def string_no_spaces(self, msg: str = "Enter: ") -> str:
        correct = False
        value = ""
        while not correct:
            raw = input(msg)
            if len(raw) == 0:
                print("Invalid input: No input")
                continue
            if " " in raw:
                print('Input contains a blank space. Please use "_" instead to avert possible errors.')
                continue
            correct = True
            value = raw
        return value

    def yes_no(self, msg: str = "Placeholder") -> bool:
        while True:
            raw = input(msg)
            if raw in ("y", "Y"):
                return True
            if raw in ("n", "N"):
                return False
            print("Invalid input: Only y/Y (yes) and n/N (no) accepted")

    def overwrite_or_rename(self, msg: str = 'Enter "o" to overwrite or "r" to rename:') -> bool:
        while True:
            raw = input(msg)
            if raw in ("o", "O"):
                return True
            if raw in ("r", "R"):
                return False
            print('Invalid input: Only o/O (letter) and r/R  accepted')
