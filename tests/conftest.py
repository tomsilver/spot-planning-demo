"""Shared configurations for pytest.

See https://docs.pytest.org/en/6.2.x/fixture.html.
"""


def pytest_addoption(parser):
    """Enable a command line flag for running tests decorated with @runreal."""
    parser.addoption(
        "--runreal",
        action="store_true",
        dest="runreal",
        default=False,
        help="Run tests with real robot",
    )
