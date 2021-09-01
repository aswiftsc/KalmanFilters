import pygame
import matplotlib.pyplot as plt
from kalman import *
from math import sin, cos, sqrt
from time import sleep


# seed simulation
np.random.seed(24)
(A, B, H, Q, R) = create_model_parameters()

# initial state
K = 30
x0 = np.array([0, -0.1, 0, 0.1])
u = np.zeros((K,4))
#u[:,1] = -0.01
u[:,3] = -0.03
prior = 0 * np.eye(4)

# simulate system
(state, meas) = simulate_system(K, x0, u)
kalman_filter = KalmanFilter(A, B, H, Q, R, x0, u[0], prior)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))
prior_state = np.zeros((K, 4))
prior_cov = np.zeros((K, 4, 4))
expmeas = np.zeros((K, 2))
for k in range(K):
    kalman_filter.predict()
    (x_prior, P_prior) = kalman_filter.get_state()
    expmeas[k, :] = kalman_filter.get_expmeas()
    if k+1 == len(u):      
      kalman_filter.update(meas[k, :], u[0])
    else:
      kalman_filter.update(meas[k, :], u[k+1])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P
    prior_state[k, :] = x_prior
    prior_cov[k, ...] = P_prior

# Initialise window
pygame.init()
screen = pygame.display.set_mode([730, 699])
myfont = pygame.font.SysFont('arial', 0) #20
background = pygame.image.load("threeplanes.png")
screen.blit(background, (0, 0))
scale = 200
origin1 = np.array([[530], [65]])
origin2 = np.array([[530], [290]])
origin3 = np.array([[530], [520]])

# define a map from (x, y)-space to the background planes
mapmatrix = np.array([[6, 1], [0, 2]])
def planemap(x, y):
    v = np.array([[x], [y]])
    return np.dot(mapmatrix, v)
    
class Point():
    def __init__(self, v, colour, scale, origin):
        self.v = v
        self.colour = colour
        self.scale = scale
        self.origin = origin
        self.draw()
        
    def draw(self):
        w = np.add(scale*np.array([[self.v[0]], [self.v[1]]]), self.origin)
        pygame.draw.circle(screen, self.colour, (w[0,0], w[1,0]), 3)
    
    def rescale(self, scale):
        self.scale = scale
        
class Covpoint():
    def __init__(self, v, colour, scale, origin, stdev=0.01):
        self.v = v
        self.radius = 20*stdev
        self.colour = colour
        self.scale = scale
        self.origin = origin
        self.draw()
        
    def draw(self):
        w = np.add(scale*np.array([[self.v[0]], [self.v[1]]]), self.origin)
        centre = np.array([[w[0,0]], [w[1,0]]])
        pygame.draw.circle(screen, self.colour, (w[0,0], w[1,0]), 3)
        p1 = np.add(self.radius*self.scale*planemap(1, 0), centre)
        if self.radius > 0:
            for i in range(1,80):
                p2 = np.add(self.radius*self.scale*planemap(cos((i+1)*3.1416/40), sin((i+1)*3.1416/40)), centre)
                pygame.draw.line(screen, self.colour, (p1[0,0], p1[1,0]), (p2[0,0], p2[1,0]), 2)
                p1 = p2
    
    def rescale(self, scale):
        self.scale = scale

def draw_line_dashed(colour, start_pos, end_pos, width=1, dash_length=3, exclude_corners=True):
    # convert tuples to numpy arrays
    start_pos = np.array(start_pos)
    end_pos   = np.array(end_pos)

    # get euclidian distance between start_pos and end_pos
    length = np.linalg.norm(end_pos - start_pos)

    # get amount of pieces that line will be split up in (half of it are amount of dashes)
    dash_amount = int(length / dash_length)

    # x-y-value-pairs of where dashes start (and on next, will end)
    dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()

    return [pygame.draw.line(screen, colour, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width)
            for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)]
    
#implement the visualisation in pygame:
clock = pygame.time.Clock()
running = True
k = 0
permanents = []
while running:
    # Check if window has been closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # check pressed keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        sleep(0.2)
        
        if k % 6 == 0:
        
            # add the first estimated point and ground-truth point (assume identical, stdev = 0)
            if k == 0:                
                w = planemap(x0[0], x0[2])
                permanents.append(Point((w[0,0], w[1,0]), (10, 10, 40), scale, origin3))
                textsurface = myfont.render('x0', False, (0, 0, 0))
                w = np.add(scale*w, origin3)
                screen.blit(textsurface,(w[0,0]-35, w[1,0]-20))
                
                w = planemap(x0[0], x0[2])
                permanents.append(Covpoint((w[0,0], w[1,0]), (40, 10, 10), scale, origin1, 0))
                textsurface = myfont.render('X0|0', False, (0, 0, 0))
                w = np.add(scale*w, origin1)
                screen.blit(textsurface,(w[0,0]-35, w[1,0]-20))
                
                prevest = (w[0,0], w[1,0])
            
            # add the estimated point and clear previous priors + expected measurements
            else:
                # clear screen and draw permanents
                screen.blit(background, (0, 0))
                for element in permanents:                   
                    element.draw()
                    
                w = planemap(est_state[int(k/6-1), 0], est_state[int(k/6-1), 2])
                permanents.append(Covpoint((w[0,0], w[1,0]), (40, 10, 10), scale, origin1, stdev))
                textsurface = myfont.render('X' + str(int(k/6)) + '|' + str(int(k/6)), False, (0, 0, 0))
                w = np.add(scale*w, origin1)
                screen.blit(textsurface,(w[0,0]-35, w[1,0]-20))    
                
                prevest = (w[0,0], w[1,0])
                
        elif k % 6 == 1:
            # check if next ground-truth point goes over the edge
            new_x = scale*state[int((k-1)/6), 0]
            new_y = scale*state[int((k-1)/6), 2]
            if new_x < -70 or new_x > 28 or new_y < -53 or new_y > 69:
                scale = scale*0.7
                screen.blit(background, (0, 0))
                for element in permanents:                   
                    element.rescale(scale)
                    element.draw()
                
                # update previous estimate
                w = np.add(scale*planemap(est_state[int((k-7)/6), 0], est_state[int((k-7)/6), 2]), origin1)
                prevest = (w[0,0], w[1,0])
                
            
            # add the next ground-truth point
            w = planemap(state[int((k-1)/6), 0], state[int((k-1)/6),2])
            permanents.append(Point((w[0,0], w[1,0]), (10, 10, 40), scale, origin3))
            textsurface = myfont.render('x' + str(int((k+5)/6)), False, (0, 0, 0))
            w = np.add(scale*w, origin3)
            screen.blit(textsurface,(w[0,0]-35, w[1,0]-20))
            
            prevtrue = (w[0,0], w[1,0])
            
        elif k % 6 == 2: 
            
            # apply the motion model to get the prior estimate
            w = planemap(prior_state[int((k-2)/6), 0], prior_state[int((k-2)/6), 2])
            stdev = np.mean([prior_cov[int((k-2)/6), 0, 0], prior_cov[int((k-2)/6), 2, 2]]) # take the mean of x-x covariance and y-y covariance, for simplicity
            Covpoint((w[0,0], w[1,0]), (40, 10, 255), scale, origin1, stdev)
            textsurface = myfont.render('X' + str(int((k+4)/6)) + '|' + str(int((k-2)/6)), False, (40, 10, 255))
            w = np.add(scale*w, origin1)
            screen.blit(textsurface,(w[0,0]-35, w[1,0]-20))
            
            draw_line_dashed((10, 10, 40), prevest, (w[0,0], w[1,0]))
            prevest = (w[0,0], w[1,0])
        
        elif k % 6 == 3:
            
            # find expected measurement
            w = planemap(expmeas[int((k-3)/6), 0], expmeas[int((k-3)/6), 1])
            Covpoint((w[0,0], w[1,0]), (255, 10, 40), scale, origin2, sqrt(stdev**2 + R[0,0]**2))
            textsurface = myfont.render('Y' + str(int((k+3)/6)) + '|' + str(int((k-3)/6)), False, (255, 10, 40))
            w = np.add(scale*w, origin2)
            screen.blit(textsurface,(w[0,0]-35, w[1,0]-20))
            
            draw_line_dashed((10, 10, 40), prevest, (w[0,0], w[1,0]))
            prevest = (w[0,0], w[1,0])
        
        elif k % 6 == 4:
        
            # make the measurement
            w = planemap(meas[int((k-4)/6), 0], meas[int((k-4)/6), 1])
            permanents.append(Point((w[0,0], w[1,0]), (20, 180, 20), scale, origin2))
            Covpoint((w[0,0], w[1,0]), (20, 180, 20), scale, origin2, R[0,0])
            textsurface = myfont.render('y' + str(int((k+2)/6)), False, (20, 180, 20))
            w = np.add(scale*w, origin2)
            screen.blit(textsurface,(prevtrue[0]-35, prevtrue[1]-60))
            
            draw_line_dashed((20, 180, 20), prevtrue, (w[0,0], w[1,0]))
            prevtrue = (w[0,0], w[1,0])
            
        else:
            
            # update estimate
            w = planemap(est_state[int((k-5)/6), 0], est_state[int((k-5)/6), 2])
            stdev = np.mean([est_cov[int((k-5)/6), 0, 0], est_cov[int((k-5)/6), 2, 2]]) # take the mean of x-x covariance and y-y covariance, for simplicity
            Covpoint((w[0,0], w[1,0]), (200, 140, 10), scale, origin1, stdev)
            textsurface = myfont.render('X' + str(int((k+1)/6)) + '|' + str(int((k+1)/6)), False, (200, 140, 10))
            w = np.add(scale*w, origin1)
            screen.blit(textsurface,(prevtrue[0]-35, prevtrue[1]-20))
            
            draw_line_dashed((200, 140, 10), prevest, (w[0,0], w[1,0]))
            draw_line_dashed((200, 140, 10), prevtrue, (w[0,0], w[1,0]))
           
        k += 1
        if k == K*5:
            running = False

    pygame.display.flip()
    clock.tick(20) # max framerate (fps)

pygame.quit()
