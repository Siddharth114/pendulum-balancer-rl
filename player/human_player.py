import pygame

from engine.state import SimulationState
from player.base_player import BasePlayer, Move


class HumanPlayer(BasePlayer):
    def move(self, state: SimulationState) -> Move:
        keys = pygame.key.get_pressed()
        return Move(command=keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
