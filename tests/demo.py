import time
from selenium import webdriver
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from scripts.yolo import CaptchaSolverBot as CaptchaSolverBotYOLO
from scripts.template_matching import CaptchaSolverBot as CaptchaSolverBotTM

CAPTCHA_URL = "https://www.geetest.com/en/adaptive-captcha-demo"

def run_demo():
    
    # YOLO + RL
    print("\n=== ESECUZIONE: YOLO + Movimento RL ===")
    bot_yolo_rl = CaptchaSolverBotYOLO(CAPTCHA_URL)
    bot_yolo_rl.solve_captcha(use_rl=True)

    # YOLO + Base
    print("\n=== ESECUZIONE: YOLO + Movimento Base ===")
    bot_yolo_base = CaptchaSolverBotYOLO(CAPTCHA_URL)
    bot_yolo_base.solve_captcha(use_rl=False)

    # Template Matching + RL
    print("\n=== ESECUZIONE: Template Matching + Movimento RL ===")
    bot_tm_rl = CaptchaSolverBotTM(CAPTCHA_URL)
    bot_tm_rl.solve_captcha(use_rl=True)

    # Template Matching + Base
    print("\n=== ESECUZIONE: Template Matching + Movimento Base ===")
    bot_tm_base = CaptchaSolverBotTM(CAPTCHA_URL)
    bot_tm_base.solve_captcha(use_rl=False)

if __name__ == "__main__":
    run_demo()