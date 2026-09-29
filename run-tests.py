#!/usr/bin/env python
"""
run-tests.py
============
Run every unit test in the project with a single command::

    python run-tests.py

Pass a file name to run just one suite::

    python run-tests.py unittests.ModelTests

All tests run against the *test* model and log artifacts, so running the suite
never disturbs the production models or the real logs.
"""

import os
import sys
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# The three suites the assignment asks for.
SUITES = [
    "unittests.ModelTests",
    "unittests.LoggerTests",
    "unittests.ApiTests",
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Make sure the tests run from the project root.
    os.chdir(PROJECT_ROOT)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if argv:
        # Run only what the caller asked for, e.g. "unittests.ModelTests" or
        # "unittests.ModelTests.ModelTest.test_02_models_load".
        selected = SUITES if argv[0] == "all" else argv
        for name in selected:
            suite.addTests(loader.loadTestsFromName(name))
    else:
        for name in SUITES:
            suite.addTests(loader.loadTestsFromName(name))

    print("=" * 70)
    print("running {} test(s) from: {}".format(suite.countTestCases(), ", ".join(SUITES)))
    print("=" * 70)

    started = time.time()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    elapsed = time.time() - started

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED  ({} run, {} skipped, {:.1f}s)".format(
            result.testsRun, len(result.skipped), elapsed
        ))
        return 0
    print("FAILURES: {} run, {} failures, {} errors".format(
        result.testsRun, len(result.failures), len(result.errors)
    ))
    return 1


if __name__ == "__main__":
    sys.exit(main())
