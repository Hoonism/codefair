import main_program
import detectcopy
import pygame
import time
pygame.init()

display = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
FPS = 60

img = pygame.image.load("cargreen.png")
img = pygame.transform.scale(img, (600, 600))
img2 = pygame.image.load("carred.png")
img2 = pygame.transform.scale(img2, (600, 600))
img3 = pygame.image.load("caryellow.png")
img3 = pygame.transform.scale(img3, (600, 600))
img4 = pygame.image.load("green.png")
img4 = pygame.transform.scale(img4, (200, 600))
img5 = pygame.image.load("red.png")
img5 = pygame.transform.scale(img5, (200, 600))
img6 = pygame.image.load("yellow.png")
img6 = pygame.transform.scale(img6, (200, 600))

def main():
    car_yellow_time = 6000
    init_time = 5
    yellow_time = 2000
    display.fill((220, 220, 220))
    display.blit(img, (80, 0))
    display.blit(img5, (0, 0))
    pygame.display.update()
    for i in range(5):
        if main_program.run(always=True):
            break
        pygame.time.wait(1000)
    display.blit(img3, (80, 0))
    pygame.display.update()
    start = time.time()
    res = main_program.run()
    end = time.time()
    diff = end - start
    if diff*1000 < car_yellow_time:
        print(diff*1000)
        pygame.time.wait(car_yellow_time-int(diff*1000))
    if res == 1:
        init_time *= 1.5
        init_time = int(init_time)
        print("yes")
    display.blit(img2, (80, 0))
    display.blit(img4, (0, 0))
    pygame.display.update()
    font = pygame.font.SysFont('Consolas', 180)
    while init_time > 0:
        display.blit(font.render(str(init_time), True, (0, 100, 22)), (600, 230))
        display.blit(img2, (80, 0))
        display.blit(img4, (0, 0))
        init_time -= 1
        pygame.display.update()
        display.fill((220, 220, 220))
        pygame.time.wait(1000)
    while main_program.present() == 1:
        display.blit(img6, (0, 0))
        pygame.display.update()
        pygame.time.wait(yellow_time)
    pygame.quit()
        

main()