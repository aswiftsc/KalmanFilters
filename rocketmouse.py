import pygame
from kalman import *

np.random.seed(22)

# Initialise window
pygame.init()
screen = pygame.display.set_mode([500, 500])
screen.fill((255, 255, 255)) # white background

# initialise simulation
fps = 10.0
friction = 0.1 # add friction to compensate for drift (0 to 1)
(A, B, H, Q, R) = create_model_parameters(1/fps, 2 ** 2, 2 ** 2, 4 ** 2)
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
force = (0, 0)
clock = pygame.time.Clock()
running = True
while running:
    # Check if window has been closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # if mouse is held down, produce force from prevest to mouse pointer
    if pygame.mouse.get_pressed()[0]:
        mouse = pygame.mouse.get_pos()
        force = (0.1*(-prevest[0]+mouse[0]), 0.1*(-prevest[2]+mouse[1]))
        
        # draw thrust line
        pygame.draw.line(screen, (0, 255, 0), (prevest[0], prevest[2]), (mouse[0], mouse[1]), 2)
    
    # simulation update
    u = np.array([0, force[0]-friction*x[1], 0, force[1]-friction*x[3]])
    x = motion_model(x, u)
    z = meas_model(x)

    # draw measurement
    pygame.draw.circle(screen, (0, 0, 255), (z[0], z[1]), 2)
    
    # kalman update
    kalman_filter.predict()
    kalman_filter.update(z, u)
    (est, P) = kalman_filter.get_state()

    # draw estimated pose
    pygame.draw.line(screen, (255, 0, 0), (prevest[0], prevest[2]), (est[0], est[2]), 2)

    # update display
    prevest = est
    force = (0, 0)
    pygame.display.flip()
    clock.tick(fps) # max framerate

pygame.quit()


