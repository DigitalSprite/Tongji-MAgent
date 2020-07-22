"""
Interactive game, Pygame are required.
Act like a general and dispatch your solders.
"""

import os

import magent
from renderer import PyGameRenderer
from renderer.server import PursuitServer as Server
from magent import utility
from models import buffer

if __name__ == "__main__":
    # utility.check_model('battle-game')
    PyGameRenderer().start(Server())
