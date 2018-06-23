def check_contamination(contamination, low=0., high=0.5):
    """Raise ValueError if the contamination is not valid."""

    if not low < contamination <= high:
        raise ValueError(
            f'contamination must be in (low, high] but was {contamination}'
        )
