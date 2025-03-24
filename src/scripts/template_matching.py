import numpy as np
import cv2
import re
import requests
from PIL import Image
from io import BytesIO
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from pynput.mouse import Button, Controller
from selenium.webdriver.common.action_chains import ActionChains
from stable_baselines3 import PPO
from scripts.mouse_mover import MouseMover

model = PPO.load("data/mouse_movement_model.zip")

class PuzzleCaptchaSolver:
    def __init__(self, gap, bg, output_image_path=None):
        self.gap_image = gap
        self.bg_image = bg
        self.output_image_path = output_image_path


    def remove_whitespace(self, img):
        """
        Rimuove le aree di whitespace tagliando l'immagine attorno ai pixel non uniformi.
        """
        min_x, min_y, max_x, max_y = 255, 255, 0, 0
        rows, cols, _ = img.shape
        for x in range(rows):
            for y in range(cols):
                if len(set(img[x, y])) >= 2:
                    min_x = min(x, min_x)
                    min_y = min(y, min_y)
                    max_x = max(x, max_x)
                    max_y = max(y, max_y)
        cropped = img[min_x:max_x, min_y:max_y]
        return cropped

    def apply_edge_detection(self, img):
        """
        Applica l'edge detection usando il metodo Canny.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb

    def find_position_of_slide(self, slide_img, background_img):
        """
        Trova la posizione del pezzo del puzzle nel background.
        """
        # Esegue il template matching
        result = cv2.matchTemplate(background_img, slide_img, cv2.TM_CCOEFF_NORMED)

        # Trova la posizione migliore
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        img_height, img_width = background_img.shape[:2]
        tpl_height, tpl_width = slide_img.shape[:2]

        # Definisce i punti del rettangolo
        tl = max_loc
        br = (tl[0] + tpl_width, tl[1] + tpl_height)

        if self.output_image_path:
            debug_img = background_img.copy()

            cv2.rectangle(debug_img, tl, br, (0, 0, 255), 2)

            mid_x = tl[0] + tpl_width // 2
            mid_y = tl[1] + tpl_height // 2

            cv2.line(debug_img, (mid_x, 0), (mid_x, img_height), (0, 255, 0), 2)
            cv2.line(debug_img, (0, mid_y), (img_width, mid_y), (0, 255, 0), 2)

            cv2.imwrite(self.output_image_path, debug_img)
            
            img_pil = Image.open("data/processed/result_tm.png")
            img_pil.show()

        return tl[0]

    def discern(self):
        """
        Rimuove il whitespace, applica l'edge detection e trova la posizione del pezzo
        all'interno dell'immagine di background. Inoltre, salva le immagini processate.
        """
        gap_img = self.remove_whitespace(self.gap_image)
        gap_edges = self.apply_edge_detection(gap_img)
        bg_edges = self.apply_edge_detection(self.bg_image)

        # Salva le immagini processate
        cv2.imwrite("data/processed/processed_piece.png", gap_edges)
        cv2.imwrite("data/processed/processed_bg.png", bg_edges)

        slide_position = self.find_position_of_slide(gap_edges, bg_edges)
        return slide_position

class CaptchaSolverBot:
    def __init__(self, captcha_url):
        self.driver = webdriver.Chrome()
        self.driver.get(captcha_url)
        self.mouse = Controller()
        self.mouse_mover = MouseMover(self.driver)

    def capture_captcha_images(self):
        """
        Cattura lo screenshot del captcha e restituisce i file contenenti la 'piece' e il background.
        Nota: occorre implementare la logica per separare le due immagini dallo screenshot.
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

        piece = self.driver.find_element(By.CLASS_NAME, "geetest_slice_bg")
        png_data = piece.screenshot_as_png
        image = Image.open(BytesIO(png_data))
        img_np = np.array(image)
        piece_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

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

    def solve_captcha(self, use_rl):
        """
        Risolve il captcha estraendo l'immagine, trovando la posizione e muovendo il mouse.
        """
        gap_image, bg_image = self.capture_captcha_images()
        
        solver = PuzzleCaptchaSolver(
            gap=gap_image,
            bg=bg_image,
            output_image_path="data/processed/result_tm.png"
        )

        result = solver.discern()

        if result is not None:
            print(f"La posizione del puzzle è: {result}")

            if use_rl:
                print("Utilizzando movimento con Reinforcement Learning...")
                self.mouse_mover.move_slider_with_rl(result)
            else:
                print("Utilizzando movimento base...")
                self.mouse_mover.move_slider_yolo(result)

            sleep(2)
            self.driver.quit
        else:
            print("Errore: non è stata rilevata alcuna posizione valida per il captcha.")



if __name__ == "__main__":
    bot = CaptchaSolverBot("https://www.geetest.com/en/adaptive-captcha-demo")
    bot.solve_captcha()
