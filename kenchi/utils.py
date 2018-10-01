def check_contamination(contamination, low=0., high=0.5):
    """Raise ValueError if the contamination is not valid."""

    if contamination != 'auto' and not low < contamination <= high:
        raise ValueError(
            f'contamination must be in (low, high] but was {contamination}'
        )


def check_novelty(novelty, method):
    """Raise AttributeError if ``novelty`` is not valid."""

    if novelty and method == 'fit_predict':
        raise AttributeError(
            f'{method} is not available when novelty=True, use '
            f'novelty=False if you want to predict on the training data'
        )

    elif not novelty and method != 'fit_predict':
        raise AttributeError(
            f'{method} is not available when novelty=False, use '
            f'novelty=True if you want to predict on new unseen data'
        )
