"""
    This is example code for creating a grid using pygame
    Credit to dannytpeck for the following code
    I do not take credit for this code
"""
import pygame

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.init()

# Set the width and height of the screen [width, height]
size = (255, 255)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

width = 20
height = 20
margin = 5

grid = [[0 for x in range(10)] for y in range(10)]

grid[1][5] = 1

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif pygame.mouse.get_pressed()[0]: #event.type == pygame.MOUSEBUTTONDOWN:
            column = pos[0] // (width + margin)
            row = pos[1] // (height + margin)
            # Debug prints
            print("Click ", pos, "Grid coordinates: ", row, column)
            grid[row][column] = 1

    # --- Game logic should go here
    pos = pygame.mouse.get_pos()
    x = pos[0]
    y = pos[1]

    # --- Drawing code should go here

    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(BLACK)

    for row in range(10):
        for column in range(10):
            if grid[row][column] == 1:
                color = GREEN
            else:
                color = WHITE
            pygame.draw.rect(screen, color,
                             [margin + (margin + width) * column, margin + (margin + height) * row, width, height])

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit()