from bdd_helper import *
from c01_intro import *
import os
import pytest

run_heavy_test = bool(os.environ.get('RUN_HEAVY_TEST', "False"))


@pytest.mark.skipif(not run_heavy_test, reason="Run_heavy_test")
def test_tune4_cat_dog():
    Given("default parameters")

    When("train and classify")
    tune4_cat_dog()
    is_cat, _ = classifiy_cat_dog()

    Then("it is expected")
    assert is_cat
