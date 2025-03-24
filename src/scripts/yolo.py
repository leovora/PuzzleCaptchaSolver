import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import re
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from mouse_mover import MouseMover
from ultralytics import YOLO

class CaptchaSolverBot:
    def __init__(self, captcha_url):
        self.driver = webdriver.Chrome()
        self.driver.get(captcha_url)
        self.mouse_mover = MouseMover(self.driver)  
        self.model = YOLO(r"data/multi_cls.onnx", task="detect")


    def capture_captcha_images(self):
        """
        Cattura lo screenshot del captcha e restituisce i file contenenti la 'piece' e il background.
        """
        sleep(1)
        try:
            selection_btn = self.driver.find_element(By.CLASS_NAME, "tab-item-1")
            selection_btn.click()
            sleep(1)
            btn = self.driver.find_element(By.CLASS_NAME, "geetest_btn_click")
            btn.click()
            sleep(3)
        except Exception as e:
            raise Exception("CAPTCHA button not found!") from e

        # Ottieni immagine del pezzo
        piece = self.driver.find_element(By.CLASS_NAME, "geetest_slice_bg")
        png_data = piece.screenshot_as_png
        image = Image.open(BytesIO(png_data))
        img_np = np.array(image)
        piece_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Ottieni immagine di background
        background = self.driver.find_element(By.CLASS_NAME, "geetest_bg")
        style_attribute = background.get_attribute("style")
        match = re.search(r'url\("(.+?)"\)', style_attribute)
        image_url = match.group(1)
        response = requests.get(image_url)
        png_data = response.content
        image = Image.open(BytesIO(png_data))
        img_np = np.array(image)
        background_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return piece_img, background_img

    def get_captcha_position(self, image):
        """
        Utilizza un modello AI per rilevare la posizione del captcha sulla schermata fornita.
        """
        results = self.model.predict(
            source=image,
            device='cpu',
            conf=0.8,
            imgsz=[416, 416],
        )

        if not len(results):
            return None

        # Trova la box con la massima confidenza
        box_with_max_conf = max(results, key=lambda x: x.boxes.conf.max())

        # Ottiene le coordinate della bounding box
        box_with_conf = box_with_max_conf.boxes.data.tolist()
        x_min, y_min, x_max, y_max, *_ = box_with_conf[0]

        print(f"Box rilevata con confidenza: {box_with_max_conf.boxes.conf.max()}")
        print(f"Coordinata X min: {x_min}, Coordinata X max: {x_max}")

        debug_img = image.copy()

        cv2.rectangle(debug_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        mid_x = int((x_min + x_max) / 2)
        mid_y = int((y_min + y_max) / 2)

        cv2.line(debug_img, (mid_x, 0), (mid_x, debug_img.shape[0]), (0, 255, 0), 2)
        cv2.line(debug_img, (0, mid_y), (debug_img.shape[1], mid_y), (0, 255, 0), 2)

        cv2.imwrite("data/processed/results_yolo.png", debug_img)

        img_pil = Image.open("data/processed/results_yolo.png")
        img_pil.show()
        

        return x_min

    def solve_captcha(self, use_rl):
        """
        Risolve il captcha estraendo l'immagine, trovando la posizione e muovendo il mouse.
        """
        piece_img, bg_img = self.capture_captcha_images()
        result = self.get_captcha_position(bg_img)

        if result is not None:
            print(f"La posizione del puzzle è: {result}")

            if use_rl:
                print("Utilizzando movimento con Reinforcement Learning...")
                self.mouse_mover.move_slider_with_rl(result, False)
            else:
                print("Utilizzando movimento base...")
                self.mouse_mover.move_slider(result, True)

            sleep(2)
            self.driver.quit()
        else:
            print("Errore: non è stata rilevata alcuna posizione valida per il captcha.")


if __name__ == "__main__":
    bot = CaptchaSolverBot("https://www.geetest.com/en/adaptive-captcha-demo")
    bot.solve_captcha(True)