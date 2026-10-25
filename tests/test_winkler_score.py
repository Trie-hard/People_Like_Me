from src.validation.winkler_score import winkler_score


def test_winkler_inside_interval_is_width_only():
    assert winkler_score(4.0, 6.0, 5.0, 0.9) == 2.0


def test_winkler_penalty_below():
    score = winkler_score(5.0, 7.0, 4.0, 0.9)
    assert score > 2.0
