import numpy as np
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from stable_baselines3 import PPO


class MouseMover:
    def __init__(self, driver):
        self.driver = driver
        self.model = PPO.load("data/mouse_movement_model.zip")

    def move_slider_tm(self, distance):
        """
        Muove il cursore sulla barra del captcha con un movimento lineare.
        """
        slider = self.driver.find_element(By.CLASS_NAME, "geetest_btn")
        actions = ActionChains(self.driver)
        actions.click_and_hold(slider).pause(0.2)
        actions.move_by_offset(distance, 0)
        actions.release().perform()
        
    def move_slider_yolo(self, distance):
        """
        Muove il cursore sulla barra del captcha con un movimento lineare.
        """
        slider = self.driver.find_element(By.CLASS_NAME, "geetest_btn")
        actions = ActionChains(self.driver)
        actions.click_and_hold(slider).pause(0.2)
        actions.move_by_offset(distance - 15, 0)
        actions.release().perform()

    def move_slider_with_rl(self, distance):
        """
        Muove il cursore utilizzando un modello RL per un movimento più naturale.
        """
        if self.model is None:
            raise ValueError("Modello RL non caricato.")

        slider = self.driver.find_element(By.CLASS_NAME, "geetest_btn")
        actions = ActionChains(self.driver)
        x_start = slider.location['x']
        target_x = x_start + distance

        actions.click_and_hold(slider).pause(0.2)

        obs = np.array([0.5, 0.5]) 
        scaling_factor = 5

        while True:
            action, _ = self.model.predict(obs)  
            dx, _ = action  
            
            dx = max(dx * scaling_factor, 0)
            
            remaining_distance = target_x - x_start
            
            dx = min(dx, remaining_distance)  

            actions.move_by_offset(dx, 0)
            actions.perform()
            x_start += dx

            if remaining_distance < 2:
                break

        actions.release().perform()