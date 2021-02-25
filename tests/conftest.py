from pathlib import Path
import os

import pytest



TEST_ASSETS_DIR = Path(__file__).parent / 'assets'


###############################################################################
# Pytest fixtures
###############################################################################



@pytest.fixture(scope='session')
def root_dir():
    """preload example audio"""
    return TEST_ASSETS_DIR