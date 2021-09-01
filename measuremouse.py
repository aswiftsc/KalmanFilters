import pygame
from kalman import *

np.random.seed(22)

# Initialise window
pygame.init()
screen = pygame.display.set_mode([500, 500])
screen.fill((255, 255, 255)) # white background

# initialise simulation
fps = 40.0
(A, B, H, Q, R) = create_model_parameters(1/fps, 200 ** 2, 200 ** 2, 4 ** 2) # sx, sy need to be >3 to avoid sever lag??? explain overshot problem!!!
motion_model = MotionModel(A, B, Q)
meas_model = MeasurementModel(H, R)
(m, _) = Q.shape
(n, _) = R.shape
prior = 0 * np.eye(4)
x0 = np.array([0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])

# initialise kalman filter
kalman_filter = KalmanFilter(A, B, H, Q, R, x0, u0, prior)

# Run until the user asks to quit
x = x0
prevest = x0
clock = pygame.time.Clock()
running = True
while running:
    # Check if window has been closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # simulation update
    mouse = pygame.mouse.get_pos()
    x = motion_model(x, u0)
    z = meas_model(np.array([mouse[0], x[1], mouse[1], x[3]]))

    # draw measurement
    pygame.draw.circle(screen, (0, 0, 255), (z[0], z[1]), 2)
    
    # kalman update
    kalman_filter.predict()
    kalman_filter.update(z, u0)
    (est, P) = kalman_filter.get_state()

    # draw estimated pose
    pygame.draw.line(screen, (255, 0, 0), (prevest[0], prevest[2]), (est[0], est[2]), 2)

    # update display
    prevest = est
    pygame.display.flip()
    clock.tick(fps) # max framerate

pygame.quit()


