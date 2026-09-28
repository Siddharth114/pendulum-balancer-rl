import pygame

pygame.init()

WINDOW_HEIGHT = 600
WINDOW_WIDTH = 800

WINDOW_BACKGROUND_COLOR = (222, 222, 209)

CONTAINER_RECTANGLE_TOP = 290
CONTAINER_RECTANGLE_LEFT = 150

CONTAINER_RECTANGLE_WIDTH = 500
CONTAINER_RECTANGLE_HEIGHT = 20
CONTAINER_RECTANGLE_COLOR = (197, 199, 188)
CONTAINER_RECTANGLE = pygame.Rect(
    CONTAINER_RECTANGLE_LEFT,
    CONTAINER_RECTANGLE_TOP,
    CONTAINER_RECTANGLE_WIDTH,
    CONTAINER_RECTANGLE_HEIGHT,
)

DRAGGABLE_PENDULUM_CIRCLE_RADIUS = 15
DRAGGABLE_PENDULUM_CIRCLE_INITIAL_LEFT = WINDOW_WIDTH // 2
DRAGGABLE_PENDULUM_CIRCLE_INITIAL_TOP = WINDOW_HEIGHT // 2
DRAGGABLE_PENDULUM_CIRCLE_COLOR = (182, 174, 159)
DRAGGABLE_PENDULUM_CIRCLE_DRAGGING = False

DRAGGABLE_CIRCLE = pygame.Rect(
    DRAGGABLE_PENDULUM_CIRCLE_INITIAL_LEFT - DRAGGABLE_PENDULUM_CIRCLE_RADIUS,
    DRAGGABLE_PENDULUM_CIRCLE_INITIAL_TOP - DRAGGABLE_PENDULUM_CIRCLE_RADIUS,
    DRAGGABLE_PENDULUM_CIRCLE_RADIUS * 2,
    DRAGGABLE_PENDULUM_CIRCLE_RADIUS * 2,
)

PENDULUM_ARM_RECTANGLE_WIDTH = 10
PENDULUM_ARM_RECTANGLE_HEIGHT = 100
PENDULUM_ARM_RECTANGLE_COLOR = (82, 70, 70)

PENDULUM_ARM_RECTANGLE = pygame.Rect(
    DRAGGABLE_PENDULUM_CIRCLE_INITIAL_LEFT - PENDULUM_ARM_RECTANGLE_WIDTH // 2,
    DRAGGABLE_PENDULUM_CIRCLE_INITIAL_TOP,
    PENDULUM_ARM_RECTANGLE_WIDTH,
    PENDULUM_ARM_RECTANGLE_HEIGHT,
)


PENDULUM_HEAD_CIRCLE_RADIUS = 10
PENDULUM_HEAD_CIRCLE_INITIAL_LEFT = DRAGGABLE_PENDULUM_CIRCLE_INITIAL_LEFT
PENDULUM_HEAD_CIRCLE_INITIAL_TOP = (
    DRAGGABLE_PENDULUM_CIRCLE_INITIAL_TOP
    + PENDULUM_ARM_RECTANGLE_HEIGHT
    - PENDULUM_HEAD_CIRCLE_RADIUS
)
PENDULUM_HEAD_COLOR = (182, 174, 159)
PENDULUM_HEAD_CIRCLE = pygame.Rect(
    PENDULUM_HEAD_CIRCLE_INITIAL_LEFT - PENDULUM_HEAD_CIRCLE_RADIUS,
    PENDULUM_HEAD_CIRCLE_INITIAL_TOP,
    PENDULUM_HEAD_CIRCLE_RADIUS * 2,
    PENDULUM_HEAD_CIRCLE_RADIUS * 2,
)


draggable_circle_offset_x = 0


screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Double Pendulum RL")


clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and DRAGGABLE_CIRCLE.collidepoint(event.pos):
                DRAGGABLE_PENDULUM_CIRCLE_DRAGGING = True
                mouse_x, mouse_y = event.pos
                draggable_circle_offset_x = DRAGGABLE_CIRCLE.x - mouse_x

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                DRAGGABLE_PENDULUM_CIRCLE_DRAGGING = False

        elif event.type == pygame.MOUSEMOTION and DRAGGABLE_PENDULUM_CIRCLE_DRAGGING:
            mouse_x, mouse_y = event.pos
            DRAGGABLE_CIRCLE.x = max(
                min(
                    mouse_x + draggable_circle_offset_x,
                    CONTAINER_RECTANGLE_LEFT
                    + CONTAINER_RECTANGLE_WIDTH
                    - DRAGGABLE_CIRCLE.width,
                ),
                CONTAINER_RECTANGLE_LEFT,
            )
            PENDULUM_ARM_RECTANGLE.x = (
                DRAGGABLE_CIRCLE.x
                + DRAGGABLE_CIRCLE.width // 2
                - PENDULUM_ARM_RECTANGLE_WIDTH // 2
            )
            PENDULUM_HEAD_CIRCLE.x = (
                DRAGGABLE_CIRCLE.x + PENDULUM_HEAD_CIRCLE_RADIUS // 2
            )

    screen.fill(WINDOW_BACKGROUND_COLOR)

    pygame.draw.rect(
        surface=screen,
        color=CONTAINER_RECTANGLE_COLOR,
        rect=CONTAINER_RECTANGLE,
        border_radius=10,
    )

    pygame.draw.rect(
        surface=screen,
        color=PENDULUM_ARM_RECTANGLE_COLOR,
        rect=PENDULUM_ARM_RECTANGLE,
        border_radius=5,
    )

    pygame.draw.rect(
        surface=screen,
        color=DRAGGABLE_PENDULUM_CIRCLE_COLOR,
        rect=DRAGGABLE_CIRCLE,
        border_radius=DRAGGABLE_PENDULUM_CIRCLE_RADIUS,
    )

    pygame.draw.rect(
        surface=screen,
        color=PENDULUM_HEAD_COLOR,
        rect=PENDULUM_HEAD_CIRCLE,
        border_radius=PENDULUM_HEAD_CIRCLE_RADIUS,
    )

    pygame.display.flip()
    clock.tick(60)


pygame.quit()
